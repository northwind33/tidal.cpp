#include "internal.hpp"
#include "sampling.hpp"
#include "prefill_tensor.hpp"
#include "prefill_chunks.hpp"
#include "kv_export.hpp"
#include "profiler.hpp"
#include "ggml-cpu.h"
#ifdef TIDAL_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <numeric>

namespace tidal {
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
Context context(size_t bytes) {
    Context result(ggml_init({bytes, nullptr, true}), ggml_free);
    if (!result) throw std::runtime_error("Cannot allocate ggml context");
    return result;
}
Backend make_backend(const Options & opts) {
    if (opts.kv_placement != KVPlacement::CPU && opts.kv_placement != KVPlacement::GPU)
        throw std::invalid_argument("Invalid KV placement");
    if (opts.kv_placement == KVPlacement::GPU && opts.backend != "cuda")
        throw std::invalid_argument("GPU KV placement requires --backend cuda; no CPU fallback");
    if (opts.threads < 1 || opts.threads > 256 || opts.device < 0)
        throw std::invalid_argument("Invalid thread count or CUDA device index");
    ggml_backend_t raw = nullptr;
    if (opts.backend == "cpu") {
        raw = ggml_backend_cpu_init();
        if (raw) ggml_backend_cpu_set_n_threads(raw, opts.threads);
    } else if (opts.backend == "cuda") {
#ifdef TIDAL_CUDA
        if (opts.device >= ggml_backend_cuda_get_device_count())
            throw std::runtime_error("Requested CUDA device is unavailable; no CPU fallback");
        raw = ggml_backend_cuda_init(opts.device);
#else
        throw std::runtime_error("CUDA was not compiled: rebuild with -DTIDAL_CUDA=ON, or explicitly use --backend cpu for reference testing");
#endif
    } else {
        throw std::invalid_argument("Backend must be cuda or cpu");
    }
    if (!raw) throw std::runtime_error("Failed to initialize " + opts.backend + "; no CPU fallback");
    return Backend(raw, ggml_backend_free);
}

// Each phase owns its graph storage. All nodes go to one backend; no scheduler
// capable of silently moving an unsupported GPU operation to the CPU is used.
class Graph {
public:
    Profiler & profiler;
    std::string phase, backend_label;
    int position, layer;
    ProfileScope build_scope;
    Context ctx;
    Buffer storage{nullptr, ggml_backend_buffer_free};
    ggml_cgraph * graph;
    ggml_backend_t backend;
    PrefillMemory * memory = nullptr;
    // Decode graphs borrow an engine-owned allocator so each token reuses one
    // device compute buffer instead of allocating and freeing per graph.
    ggml_gallocr * shared = nullptr;
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator{nullptr, ggml_gallocr_free};
    size_t allocated_bytes = 0;
    bool reserved = false;
    std::function<void()> before_compute;
    Statistics & stats;
    bool device;
    std::vector<std::function<void()>> uploads;
    Graph(ggml_backend_t b, Statistics & s, bool d, Profiler & p,
          std::string ph, int pos, int l, PrefillMemory * mem = nullptr, ggml_gallocr * alloc = nullptr, size_t nodes = 2048)
        : profiler(p), phase(std::move(ph)), backend_label(ggml_backend_name(b)), position(pos), layer(l),
          build_scope(&p, "host", "graph_build", phase, pos, l, "CPU"),
          ctx(context(std::max<size_t>(4, nodes / 512) * 1024 * 1024)),
          graph(ggml_new_graph_custom(ctx.get(), nodes, false)),
          backend(b), memory(mem), shared(alloc), stats(s), device(d) {}
    ~Graph() {
        ProfileScope release(&profiler, "host", "graph_free", phase, position, layer, "CPU");
        storage.reset();
        allocator.reset();
        if (memory && allocated_bytes) memory->release(allocated_bytes);
        ctx.reset();
    }
    ggml_context * c() { return ctx.get(); }
    void add(ggml_tensor * t) { if (memory) ggml_set_output(t); ggml_build_forward_expand(graph, t); }
    // Read back after run(): a shared allocator may otherwise reuse its memory.
    void output(ggml_tensor * t) { ggml_set_output(t); ggml_build_forward_expand(graph, t); }
    bool allocated_inputs() const { return memory || shared; }
    // Per-token data of a reusable decode graph. Their values change between
    // runs while the graph topology, and thus its captured CUDA graph, does not.
    ggml_tensor * token_input = nullptr, * position_input = nullptr, * kv_row_input = nullptr, * mask_input = nullptr;
    ggml_tensor * finite = nullptr;
    int kv_padded = 0;
    ggml_tensor * input(ggml_type type, int64_t n) {
        auto * t = ggml_new_tensor_1d(c(), type, n);
        ggml_set_input(t);
        return t;
    }
    ggml_tensor * integer(int32_t value) {
        return integers({value});
    }
    ggml_tensor * integers(const std::vector<int32_t> & values) {
        auto * t = ggml_new_tensor_1d(c(), GGML_TYPE_I32, int64_t(values.size()));
        if (allocated_inputs()) ggml_set_input(t);
        uploads.emplace_back([this, t, values] { put(t, values.data(), values.size() * sizeof(int32_t)); });
        return t;
    }
    ggml_tensor * floats(const std::vector<float> & values) {
        auto * t = ggml_new_tensor_1d(c(), GGML_TYPE_F32, int64_t(values.size()));
        if (allocated_inputs()) ggml_set_input(t);
        uploads.emplace_back([this, t, values] { put(t, values.data(), values.size() * sizeof(float)); });
        return t;
    }
    void put(ggml_tensor * t, const void * data, size_t size) { put_at(t, data, 0, size); }
    void put_at(ggml_tensor * t, const void * data, size_t offset, size_t size) {
        // Graph allocators skip inputs whose consumers were never expanded.
        if (!t->buffer && !t->data) return;
        ProfileScope transfer(&profiler, "transfer", device ? "H2D" : "host_write", phase, position, layer, backend_label);
        transfer.event.bytes = size;
        auto start = Clock::now();
        ggml_backend_tensor_set(t, data, offset, size);
        stats.transfer_ms += elapsed(start);
        if (device) stats.host_to_device_bytes += size;
    }
    template<class T> std::vector<T> get(ggml_tensor * t) {
        std::vector<T> values(size_t(ggml_nelements(t)));
        if (values.size() * sizeof(T) != ggml_nbytes(t) || !ggml_is_contiguous(t))
            throw std::logic_error("Readback must be contiguous with matching type");
        ProfileScope transfer(&profiler, "transfer", device ? "D2H" : "host_read", phase, position, layer, backend_label);
        transfer.event.bytes = ggml_nbytes(t);
        auto start = Clock::now();
        ggml_backend_tensor_get(t, values.data(), 0, ggml_nbytes(t));
        stats.transfer_ms += elapsed(start);
        if (device) stats.device_to_host_bytes += ggml_nbytes(t);
        return values;
    }
    void reserve() {
        if (reserved) return;
        build_scope.finish();
        ProfileScope validation(&profiler, "host", "supports_op", phase, position, layer, "CPU");
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto * node = ggml_graph_node(graph, i);
            if (!ggml_backend_supports_op(backend, node))
                throw std::runtime_error(std::string(ggml_backend_name(backend)) + " does not support " +
                                         ggml_op_name(node->op) + "; no CPU fallback");
        }
        validation.finish();
        ProfileScope allocation(&profiler, "host", "graph_allocate", phase, position, layer, "CPU");
        if (memory) {
            allocator.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)));
            if (!allocator) throw std::runtime_error("Cannot create prefill graph allocator");
            size_t bytes = 0;
            ggml_gallocr_reserve_n_size(allocator.get(), graph, nullptr, nullptr, &bytes);
            memory->acquire(bytes); allocated_bytes = bytes;
            if (!ggml_gallocr_reserve(allocator.get(), graph)) throw PrefillAllocationError("Insufficient VRAM for prefill graph reservation");
            if (!ggml_gallocr_alloc_graph(allocator.get(), graph)) throw PrefillAllocationError("Insufficient VRAM for prefill graph allocation");
        } else if (shared) {
            // The previous decode graph has completed (run() synchronizes), so a
            // buffer growth inside the allocator cannot free memory in use.
            if (!ggml_gallocr_alloc_graph(shared, graph)) throw std::runtime_error("Decode graph allocation failed");
        } else {
            storage.reset(ggml_backend_alloc_ctx_tensors(c(), backend));
            if (!storage) throw std::runtime_error("Graph allocation failed");
        }
        allocation.finish();
        reserved = true;
    }
    void run() {
        reserve();
        for (auto & upload : uploads) upload();
        if (before_compute) before_compute();
        ProfileScope compute(&profiler, "graph", "compute", phase, position, layer, backend_label);
        auto start = Clock::now();
        ggml_status status = GGML_STATUS_SUCCESS;
        if (profiler.mode == ProfileMode::Nodes) {
            // Reuse allocated tensors; execute each node exactly once in graph
            // order. This intentionally disables fusion and synchronizes per op.
            auto * single = ggml_new_graph_custom(c(), 1, false);
            for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                auto * node = ggml_graph_node(graph, i);
                ggml_graph_clear(single);
                ggml_graph_add_node(single, node);
                ProfileScope op(&profiler, "node", node->name[0] ? node->name : ggml_op_name(node->op),
                                phase, position, layer, backend_label);
                op.event.node = i;
                op.event.op = ggml_op_desc(node);
                op.event.dtype = ggml_type_name(node->type);
                op.event.shape.assign(node->ne, node->ne + GGML_MAX_DIMS);
                op.event.bytes = ggml_nbytes(node); // Logical output size, not measured DRAM traffic.
                for (auto * input : node->src) if (input) {
                    op.event.input_types.emplace_back(ggml_type_name(input->type));
                    op.event.input_shapes.emplace_back(input->ne, input->ne + GGML_MAX_DIMS);
                }
                op.restart();
                status = ggml_backend_graph_compute(backend, single);
                if (status != GGML_STATUS_SUCCESS) break;
            }
        } else {
            status = ggml_backend_graph_compute(backend, graph);
        }
        ggml_backend_synchronize(backend);
        stats.graph_ms += elapsed(start);
        if (status != GGML_STATUS_SUCCESS) throw std::runtime_error("ggml graph execution failed");
        stats.device_graph_nodes += ggml_graph_n_nodes(graph);
    }
};

struct LayerState {
    const PLELayer * ple = nullptr;
    ggml_tensor * ple_conv = nullptr;
    ggml_tensor * conv = nullptr;
    ggml_tensor * recurrent = nullptr;
    ggml_tensor * raw_index = nullptr;
    ggml_tensor * index_cache = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * inject = nullptr;
    std::unique_ptr<SparseAttention> attention;
    ggml_tensor * device_keys = nullptr, * device_values = nullptr;
    int device_kv_size = 0;
};
struct Mixed { ggml_tensor * value; ggml_tensor * inject; };
struct Prepared { ggml_tensor * q, * k, * v, * scores, * top; };
}

struct Engine::Impl {
    Options options;
    Backend backend;
    Model model;
    Config c;
    Statistics stats;
    Profiler profiler;
    Context state_context;
    Buffer state_buffer{nullptr, ggml_backend_buffer_free};
    ggml_tensor * hidden;
    ggml_tensor * last_logits;
    ggml_tensor * kv_token_ids = nullptr;
    uint64_t gpu_kv_bytes = 0;
    std::vector<float> captured_logits;
    std::unique_ptr<Sampler> sampler;
    std::vector<LayerState> states;
    std::vector<int32_t> previous_tokens;
    int position = 0;
    bool failed = false;
    PrefillMemory * prefill_memory = nullptr;
    Backend prefill_backend{nullptr, ggml_backend_free};
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> decode_allocator{nullptr, ggml_gallocr_free};
    // Reused while weights stay resident and the KV bucket is unchanged; prefill
    // may evict/reload weights, and reset rewinds the position.
    std::unique_ptr<Graph> decode_graph;

    Impl(const std::string & path, Options opts)
        : options(std::move(opts)), backend(make_backend(options)), model(path, backend.get()), c(model.config),
          profiler(options.profile, options.nvtx, options.profile_event_limit), state_context(context(1024 * 1024)), states(c.layers) {
        if (options.prefill_chunk_tokens < 0 || options.prefill_attention_tile < 0 || options.prefill_attention_tile > 4096 ||
            (options.prefill_precision != PrefillPrecision::FP16 && options.prefill_precision != PrefillPrecision::BF16))
            throw std::invalid_argument("Invalid prefill precision or attention tile");
        if (options.context < 0 || options.context > c.context) throw std::invalid_argument("Invalid runtime context capacity");
        if (options.context) c.context = options.context;
        auto * ctx = state_context.get();
        hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.hidden, c.branches);
        last_logits = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.vocab);
        for (int i = 0; i < c.layers; ++i) {
            auto & s = states[i];
            if (c.is_recurrent(i)) {
                const int channels = (2 * c.gdn_key_heads + c.gdn_value_heads) * c.gdn_dim;
                s.conv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.conv_kernel - 1, channels);
                s.recurrent = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                    c.gdn_dim, c.gdn_dim, c.gdn_value_heads, 1);
            } else {
                if (c.architecture == Architecture::Qwen4Exp) {
                    s.raw_index = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.index_dim, c.block_size);
                    s.index_cache = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, c.index_dim,
                                                  std::max(1, c.context / c.block_size));
                    s.inject = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.branches);
                }
                s.gate = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.q_heads * c.head_dim);
                if (options.kv_placement == KVPlacement::GPU) {
                    const auto type = options.prefill_precision == PrefillPrecision::FP16 ? GGML_TYPE_F16 : GGML_TYPE_BF16;
                    const int dim = attention_dim(), padded = (c.context+255)/256*256;
                    s.device_keys = ggml_new_tensor_3d(ctx, type, dim, padded, c.kv_heads);
                    s.device_values = ggml_new_tensor_3d(ctx, type, dim, padded, c.kv_heads);
                    gpu_kv_bytes += ggml_nbytes(s.device_keys)+ggml_nbytes(s.device_values);
                } else {
                    s.attention = std::make_unique<SparseAttention>(c.q_heads, c.kv_heads, c.head_dim,
                                                                   c.context, options.threads);
                }
            }
        }
        for (const auto & l : c.ple.layers) {
            auto & s = states[l.layer];
            s.ple = &l;
            if (c.ple.conv_kernel > 1)
                s.ple_conv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.hidden * c.branches,
                                               (c.ple.conv_kernel - 1) * c.ple.ngram_size);
        }
        if (options.kv_placement == KVPlacement::GPU) {
            if (c.architecture == Architecture::Qwen4Exp)
                kv_token_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, c.context);
            size_t free = 0, total = 0;
            ggml_backend_dev_memory(ggml_backend_get_device(backend.get()), &free, &total);
            const auto bytes = ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, ggml_backend_get_default_buffer_type(backend.get()));
            if (bytes > free)
                throw std::runtime_error("Insufficient VRAM for persistent GPU KV/state: required " +
                    std::to_string(bytes) + ", available " + std::to_string(free) +
                    "; reduce --context or choose --kv-placement cpu; no automatic CPU fallback");
        }
        state_buffer.reset(ggml_backend_alloc_ctx_tensors(ctx, backend.get()));
        if (!state_buffer) throw std::runtime_error("State allocation failed");
        sampler = std::make_unique<Sampler>(c.vocab, options.sampling, backend.get(), options.device, stats, profiler);
        reset();
    }
    Graph graph(const std::string & phase, int layer = -1, int token_offset = 0) {
        return Graph(prefill_backend ? prefill_backend.get() : backend.get(), stats, options.backend == "cuda", profiler, phase, position + token_offset, layer, prefill_memory);
    }
    // Persistent GPU KV keeps every decode operation on the device, so a token
    // runs as one graph with one synchronization on a reused compute buffer.
    bool single_graph_decode(bool capture) const { return options.kv_placement == KVPlacement::GPU && !capture; }
    ggml_gallocr * decode_allocator_get() {
        if (!decode_allocator) {
            decode_allocator.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend.get())));
            if (!decode_allocator) throw std::runtime_error("Cannot create decode graph allocator");
        }
        return decode_allocator.get();
    }
    int attention_dim() const { return c.head_dim <= 64 ? 64 : c.head_dim <= 128 ? 128 : 256; }
    // Store directly in the persistent working format: GPU FP16 KV must not
    // incur the hybrid path's intermediate BF16 rounding or host export.
    void append_device_kv(Graph & g, LayerState & s, ggml_tensor * k, ggml_tensor * v, int first, int count) {
        if (s.device_kv_size != first) throw std::logic_error("GPU KV prefix is inconsistent");
        if (g.kv_row_input && count == 1) {
            // The destination row is data, so a reused graph appends at any position.
            auto row = [&](ggml_tensor * source, ggml_tensor * target) {
                source = ggml_reshape_3d(g.c(), source, c.head_dim, 1, c.kv_heads);
                if (attention_dim() != c.head_dim) source = ggml_pad(g.c(), source, attention_dim()-c.head_dim, 0, 0, 0);
                g.add(ggml_set_rows(g.c(), target, source, g.kv_row_input));
            };
            row(k, s.device_keys); row(v, s.device_values);
            return;
        }
        auto copy = [&](ggml_tensor * source, ggml_tensor * target) {
            source = ggml_reshape_3d(g.c(), source, c.head_dim, c.kv_heads, count);
            source = ggml_cont(g.c(), ggml_permute(g.c(), source, 0, 2, 1, 3));
            source = ggml_pad(g.c(), source, attention_dim()-c.head_dim, 0, 0, 0);
            auto * destination = ggml_view_3d(g.c(), target, attention_dim(), count, c.kv_heads,
                target->nb[1], target->nb[2], size_t(first)*target->nb[1]);
            g.add(ggml_cpy(g.c(), source, destination));
        };
        copy(k, s.device_keys); copy(v, s.device_values);
    }
    ggml_tensor * flash_attention(Graph & g, ggml_tensor * q, ggml_tensor * keys, ggml_tensor * values,
                                  ggml_tensor * mask, int count, int visible) {
        // FlashAttention reads a permuted Q directly; padding is needed only
        // when the head size is below the kernel's head dimension.
        q = ggml_permute(g.c(), q, 0, 2, 1, 3);
        if (attention_dim() != c.head_dim) q = ggml_pad(g.c(), ggml_cont(g.c(), q), attention_dim()-c.head_dim, 0, 0, 0);
        // A decode graph passes one ready F16 mask shared by every layer.
        if (mask->type != GGML_TYPE_F16)
            mask = ggml_cast(g.c(), ggml_pad(g.c(), mask, 0, (64-count%64)%64, 0, 0), GGML_TYPE_F16);
        auto prefix = [&](ggml_tensor * t) {
            return ggml_view_3d(g.c(), t, attention_dim(), (visible+255)/256*256, c.kv_heads, t->nb[1], t->nb[2], 0);
        };
        auto * result = ggml_flash_attn_ext(g.c(), q, prefix(keys), prefix(values), mask,
            1.0f/std::sqrt(float(c.head_dim)), 0, 0);
        if (options.kv_placement == KVPlacement::GPU && c.architecture == Architecture::Qwen4Exp) {
            // Every causal row has at most top_blocks complete blocks plus its
            // incomplete tail. This lets capable backends gather sparse KV.
            const int64_t bound = int64_t(c.top_blocks)*c.block_size+c.block_size-1;
            ggml_flash_attn_ext_set_n_kv_max(result, int32_t(std::min<int64_t>(visible, bound)));
        }
        ggml_prec_set_acc(result, GGML_PREC_F32);
        ggml_set_name(result, ("blk."+std::to_string(g.layer)+".attention.flash").c_str());
        if (attention_dim() == c.head_dim) return result;
        return ggml_cont(g.c(), ggml_view_3d(g.c(), result, c.head_dim, c.q_heads, count,
            result->nb[1], result->nb[2], 0));
    }
    // Appends this token's KV, attends on the device and returns the layer output.
    ggml_tensor * device_attention(Graph & g, int layer, ggml_tensor * h, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                                   ggml_tensor * scores, ggml_tensor * top) {
        auto & s = states[layer];
        const int visible = position+1, padded = (visible+255)/256*256;
        const auto p = "blk."+std::to_string(layer)+".";
        append_device_kv(g, s, k, v, position, 1);
        auto * mask = g.mask_input && g.kv_padded == padded ? ggml_reshape_2d(g.c(), g.mask_input, padded, 1)
            : ggml_diag_mask_inf(g.c(), ggml_fill(g.c(), ggml_new_tensor_2d(g.c(), GGML_TYPE_F32, padded, 1), 0), position);
        if (top) {
            const int blocks = visible/c.block_size;
            std::vector<int32_t> extra(c.block_size, -1);
            for (int i=blocks*c.block_size; i<visible; ++i) extra[i-blocks*c.block_size] = i;
            auto * mapping = ggml_view_2d(g.c(), kv_token_ids, c.block_size, blocks, c.block_size*sizeof(int32_t), 0);
            mask = ggml_top_k_mask(g.c(), mask, ggml_reshape_2d(g.c(), top, top->ne[0], 1),
                ggml_reshape_2d(g.c(), scores, blocks, 1), mapping,
                ggml_reshape_2d(g.c(), g.integers(extra), c.block_size, 1));
        }
        auto * attended = flash_attention(g, q, s.device_keys, s.device_values, mask, 1, visible);
        attended = ggml_mul(g.c(), ggml_reshape_1d(g.c(), attended, c.q_heads*c.head_dim), ggml_sigmoid(g.c(), s.gate));
        auto * projected = mm(g, p+"attn.out", attended);
        auto * residual = c.architecture == Architecture::Qwen35 ? ggml_add(g.c(), h, projected)
            : combine(g, h, projected, s.inject);
        ++stats.gpu_attention_calls;
        return ffn(g, residual, p);
    }
    Selection finish_device_attention(Graph & g, int layer, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                                      ggml_tensor * scores, ggml_tensor * top, bool capture) {
        auto & s = states[layer];
        const int visible = position+1;
        if (top && capture) g.output(top);
        g.add(ggml_cpy(g.c(), device_attention(g, layer, hidden, q, k, v, scores, top), hidden));
        g.run(); s.device_kv_size = visible;
        std::vector<int32_t> selected;
        if (capture) {
            if (c.architecture == Architecture::Qwen4Exp)
                selected = expand_blocks(top ? g.get<int32_t>(top) : std::vector<int32_t>{}, visible, c.block_size);
            else { selected.resize(visible); std::iota(selected.begin(), selected.end(), 0); }
        }
        return {layer, std::move(selected)};
    }
    ggml_tensor * weight(const std::string & n) {
        auto * tensor = model.at(n);
        if (!tensor->data) throw std::logic_error("Weight group is not resident: " + n);
        return tensor;
    }
    ggml_tensor * mm(Graph & g, const std::string & n, ggml_tensor * x) {
        auto * out = ggml_mul_mat(g.c(), weight(n), x);
        // Request F32 accumulation. This does not prescribe operand conversion:
        // CPU dot types and CUDA Q8_1 activation quantization still differ.
        ggml_prec_set_acc(out, GGML_PREC_F32);
        ggml_set_name(out, n.c_str());
        return out;
    }
    ggml_tensor * norm(Graph & g, ggml_tensor * x, const std::string & n) {
        auto * rms = ggml_rms_norm(g.c(), x, c.eps);
        ggml_set_name(rms, (n + ".rms").c_str());
        auto * result = ggml_mul(g.c(), rms, weight(n));
        ggml_set_name(result, n.c_str());
        return result;
    }
    ggml_tensor * rope(Graph & g, ggml_tensor * x, int dim, int heads, int pos) {
        x = ggml_reshape_3d(g.c(), ggml_is_contiguous(x) ? x : ggml_cont(g.c(), x), dim, heads, 1);
        auto * positions = g.position_input && pos == position ? g.position_input : g.integer(pos);
        return ggml_rope_ext(g.c(), x, positions, nullptr, c.rope_dim,
                             GGML_ROPE_TYPE_NEOX, c.context, c.rope_base, 1, 0, 1, 0, 0);
    }
    Mixed mix(Graph & g, ggml_tensor * h, const std::string & p, bool inject = true) {
        auto * x = norm(g, h, p + ".norm");
        auto * flat = ggml_reshape_1d(g.c(), x, c.hidden * c.branches);
        auto * low = ggml_silu(g.c(), ggml_scale(g.c(), mm(g, p + ".down", flat), 1.0f / c.branches));
        auto * gate = ggml_sigmoid(g.c(), mm(g, p + ".up", low));
        auto * gated = ggml_mul(g.c(), flat, gate);
        auto * mixed = ggml_view_1d(g.c(), gated, c.hidden, 0);
        for (int branch = 1; branch < c.branches; ++branch)
            mixed = ggml_add(g.c(), mixed, ggml_view_1d(g.c(), gated, c.hidden, size_t(branch) * c.hidden * sizeof(float)));
        mixed = ggml_scale(g.c(), mixed, 1.0f / c.branches);
        auto * injection = inject ? ggml_scale(g.c(), ggml_sigmoid(g.c(),
            ggml_scale(g.c(), mm(g, p + ".inject", flat), 1.0f / c.branches)), 2.0f) : nullptr;
        return {mixed, injection};
    }
    ggml_tensor * combine(Graph & g, ggml_tensor * h, ggml_tensor * output, ggml_tensor * injection) {
        auto * out = ggml_reshape_2d(g.c(), output, c.hidden, 1);
        auto * repeated = ggml_repeat(g.c(), out, h);
        auto * gates = ggml_reshape_2d(g.c(), injection, 1, c.branches);
        return ggml_add(g.c(), h, ggml_mul(g.c(), repeated, gates));
    }
    ggml_tensor * ffn(Graph & g, ggml_tensor * h, const std::string & p) {
        if (c.architecture == Architecture::Qwen35) {
            auto * x = norm(g, h, p + "ffn_norm");
            // GLU form lets the CUDA backend fuse gate/up matvecs with the activation.
            auto * active = ggml_swiglu_split(g.c(), mm(g, p + "ffn.gate", x), mm(g, p + "ffn.up", x));
            return ggml_add(g.c(), h, mm(g, p + "ffn.down", active));
        }
        auto mixed = mix(g, h, p + "hc_ffn");
        if (c.experts) return combine(g, h, moe(g, mixed.value, p + "ffn."), mixed.inject);
        auto * v = ggml_swiglu_split(g.c(), mm(g, p + "ffn.gate", mixed.value), mm(g, p + "ffn.up", mixed.value));
        return combine(g, h, mm(g, p + "ffn.down", v), mixed.inject);
    }
    ggml_tensor * moe(Graph & g, ggml_tensor * x, const std::string & p) {
        auto * probabilities = ggml_soft_max(g.c(), mm(g, p + "router", x));
        auto * ids = ggml_cont(g.c(), ggml_top_k(g.c(), probabilities, c.experts_used));
        ggml_set_name(ids, (p + "expert_ids").c_str());
        auto * selected = ggml_reshape_1d(g.c(), ggml_get_rows(g.c(),
            ggml_reshape_2d(g.c(), probabilities, 1, c.experts), ids), c.experts_used);
        if (c.norm_topk) selected = ggml_div(g.c(), selected, ggml_sum_rows(g.c(), selected));
        ggml_set_name(selected, (p + "routing_weights").c_str());
        auto expert_mm = [&](const char * name, ggml_tensor * input) {
            auto * out = ggml_mul_mat_id(g.c(), weight(p + name), input, ids);
            // The pinned CUDA MUL_MAT_ID rejects GGML_PREC_F32 overrides;
            // leave the backend's native expert accumulation policy intact.
            ggml_set_name(out, (p + name).c_str());
            return out;
        };
        x = ggml_reshape_3d(g.c(), x, c.hidden, 1, 1);
        auto * active = ggml_mul(g.c(), ggml_silu(g.c(), expert_mm("gate_exps", x)), expert_mm("up_exps", x));
        auto * outputs = expert_mm("down_exps", active);
        auto * weighted = ggml_mul(g.c(), outputs, ggml_reshape_3d(g.c(), selected, 1, c.experts_used, 1));
        auto * result = ggml_view_1d(g.c(), weighted, c.hidden, 0);
        for (int i = 1; i < c.experts_used; ++i)
            result = ggml_add(g.c(), result, ggml_view_1d(g.c(), weighted, c.hidden, size_t(i) * weighted->nb[1]));
        if (c.shared_ffn) {
            auto * shared = ggml_mul(g.c(), ggml_silu(g.c(), mm(g, p + "shared.gate", x)), mm(g, p + "shared.up", x));
            shared = mm(g, p + "shared.down", shared);
            auto * gate = ggml_sigmoid(g.c(), mm(g, p + "shared_gate", x));
            result = ggml_add(g.c(), result, ggml_mul(g.c(), shared, gate));
        }
        return result;
    }
    ggml_tensor * repeat_gdn_heads(Graph & g, ggml_tensor * x) {
        // Native qwen35 GGUFs reorder V heads to ggml's tiled broadcast order.
        if (c.architecture == Architecture::Qwen35)
            return ggml_repeat_4d(g.c(), x, c.gdn_dim, c.gdn_value_heads, 1, 1);
        // ggml's fused GDN broadcasts by tiling; Qwen requires repeat_interleave.
        ggml_tensor * expanded = nullptr;
        const int repeat = c.gdn_value_heads / c.gdn_key_heads;
        for (int h = 0; h < c.gdn_value_heads; ++h) {
            auto * head = ggml_view_1d(g.c(), x, c.gdn_dim, size_t(h / repeat) * c.gdn_dim * sizeof(float));
            expanded = expanded ? ggml_concat(g.c(), expanded, head, 0) : head;
        }
        return ggml_reshape_4d(g.c(), ggml_cont(g.c(), expanded), c.gdn_dim, c.gdn_value_heads, 1, 1);
    }
    void ple_layer(int layer, int32_t token) {
        ProfileScope scope(&profiler, "layer", "ple", "ple", position, layer, ggml_backend_name(backend.get()));
        auto g = graph("ple", layer);
        g.add(ggml_cpy(g.c(), build_ple(g, layer, token, hidden), hidden));
        g.run();
    }
    ggml_tensor * build_ple(Graph & g, int layer, int32_t token, ggml_tensor * hidden) {
        auto & s = states[layer];
        ProfileScope hash(&profiler, "host", "ngram_hash", "ple", position, layer, "CPU");
        auto rows = ple_rows(c.ple, *s.ple, token, previous_tokens);
        hash.finish();
        const auto p = "blk." + std::to_string(layer) + ".ple.";
        auto * emb = ggml_get_rows(g.c(), weight(p + "embedding"), g.integers(rows));
        ggml_set_name(emb, (p + "lookup").c_str());
        emb = ggml_reshape_1d(g.c(), emb, c.ple.embedding_dim);
        auto * key = norm(g, ggml_reshape_2d(g.c(), mm(g, p + "key", emb), c.hidden, c.branches), p + "norm_key");
        auto * query = norm(g, hidden, p + "norm_query");
        auto * score = ggml_scale(g.c(), ggml_sum_rows(g.c(), ggml_mul(g.c(), key, query)),
                                  1.0f / std::sqrt(float(c.hidden)));
        auto * magnitude = ggml_sqrt(g.c(), ggml_clamp(g.c(), ggml_abs(g.c(), score), 1e-6f, INFINITY));
        auto * gate = ggml_sigmoid(g.c(), ggml_mul(g.c(), ggml_sgn(g.c(), score), magnitude));
        ggml_set_name(gate, (p + "gate").c_str());
        auto * value = ggml_repeat(g.c(), mm(g, p + "value", emb), hidden);
        auto * gated = ggml_mul(g.c(), value, gate);
        ggml_set_name(gated, (p + "gated_value").c_str());
        const int channels = c.hidden * c.branches;
        auto * normalized = ggml_reshape_2d(g.c(), norm(g, gated, p + "norm_conv"), channels, 1);
        auto * history = s.ple_conv ? ggml_concat(g.c(), s.ple_conv, normalized, 1) : normalized;
        auto * kernel = weight(p + "conv");
        ggml_tensor * convolved = nullptr;
        for (int k = 0; k < c.ple.conv_kernel; ++k) {
            auto * input = ggml_view_1d(g.c(), history, channels, size_t(k * c.ple.ngram_size) * history->nb[1]);
            auto * tap = ggml_cont(g.c(), ggml_view_2d(g.c(), kernel, 1, channels, kernel->nb[1], k * kernel->nb[0]));
            auto * term = ggml_mul(g.c(), input, ggml_reshape_1d(g.c(), tap, channels));
            ggml_set_name(term, (p + "conv.tap." + std::to_string(k)).c_str());
            convolved = convolved ? ggml_add(g.c(), convolved, term) : term;
        }
        convolved = ggml_reshape_2d(g.c(), ggml_silu(g.c(), convolved), c.hidden, c.branches);
        ggml_set_name(convolved, (p + "conv_silu").c_str());
        auto * result = ggml_add(g.c(), hidden, ggml_add(g.c(), gated, convolved));
        if (s.ple_conv) {
            // Expand the result first: the history tail overwrites the state it reads.
            ggml_build_forward_expand(g.graph, result);
            auto * tail = ggml_view_2d(g.c(), history, channels, s.ple_conv->ne[1], history->nb[1], history->nb[1]);
            g.add(ggml_cpy(g.c(), tail, s.ple_conv));
        }
        return result;
    }
    void linear_layer(int layer) {
        ProfileScope scope(&profiler, "layer", "gdn", "gdn", position, layer, ggml_backend_name(backend.get()));
        auto g = graph("gdn", layer);
        g.add(ggml_cpy(g.c(), build_linear(g, layer, hidden), hidden));
        g.run();
    }
    ggml_tensor * build_linear(Graph & g, int layer, ggml_tensor * hidden) {
        auto & s = states[layer];
        const std::string p = "blk." + std::to_string(layer) + ".";
        auto mixed = c.architecture == Architecture::Qwen35
            ? Mixed{norm(g, hidden, p + "attn_norm"), nullptr} : mix(g, hidden, p + "hc_attn");
        auto * qkv = mm(g, p + "gdn.qkv", mixed.value);
        const int kd = c.gdn_key_heads * c.gdn_dim, vd = c.gdn_value_heads * c.gdn_dim;
        const int channels = 2 * kd + vd;
        auto * row = ggml_reshape_2d(g.c(), qkv, 1, channels);
        auto * history = ggml_concat(g.c(), s.conv, row, 0);
        auto * convolved = ggml_silu(g.c(), ggml_ssm_conv(g.c(), history, weight(p + "gdn.conv")));
        ggml_set_name(convolved->src[0], (p + "gdn.convolution").c_str());
        ggml_set_name(convolved, (p + "gdn.conv_silu").c_str());
        auto * q = ggml_reshape_2d(g.c(), ggml_view_1d(g.c(), convolved, kd, 0), c.gdn_dim, c.gdn_key_heads);
        auto * k = ggml_reshape_2d(g.c(), ggml_view_1d(g.c(), convolved, kd, size_t(kd) * sizeof(float)),
                                  c.gdn_dim, c.gdn_key_heads);
        auto * v = ggml_reshape_4d(g.c(), ggml_view_1d(g.c(), convolved, vd, size_t(2 * kd) * sizeof(float)),
                                  c.gdn_dim, c.gdn_value_heads, 1, 1);
        // Qwen uses x / sqrt(sum(x*x) + eps), whereas ggml_l2_norm clamps
        // sqrt(sum) to eps. RMSNorm with eps/d followed by 1/sqrt(d) is exact.
        const float scale = 1.0f / std::sqrt(float(c.gdn_dim));
        const float l2_eps = c.architecture == Architecture::Qwen35 ? 1e-6f : c.eps;
        q = ggml_scale(g.c(), ggml_rms_norm(g.c(), q, l2_eps / c.gdn_dim), scale);
        k = ggml_scale(g.c(), ggml_rms_norm(g.c(), k, l2_eps / c.gdn_dim), scale);
        // GATED_DELTA_NET tiles key heads over value heads (head % key_heads) on
        // CPU and CUDA, which is exactly the native qwen35 layout.
        if (c.architecture == Architecture::Qwen35) {
            q = ggml_reshape_4d(g.c(), q, c.gdn_dim, c.gdn_key_heads, 1, 1);
            k = ggml_reshape_4d(g.c(), k, c.gdn_dim, c.gdn_key_heads, 1, 1);
        } else {
            q = repeat_gdn_heads(g, q); k = repeat_gdn_heads(g, k);
        }
        auto * a = ggml_softplus(g.c(), ggml_add(g.c(), mm(g, p + "gdn.a", mixed.value), weight(p + "gdn.dt_bias")));
        // qwen35 GGUF stores -exp(A_log), while the tidal schema stores A_log.
        auto * decay = c.architecture == Architecture::Qwen35 ? ggml_mul(g.c(), weight(p + "gdn.a_log"), a)
            : ggml_neg(g.c(), ggml_mul(g.c(), ggml_exp(g.c(), weight(p + "gdn.a_log")), a));
        auto * beta = ggml_sigmoid(g.c(), mm(g, p + "gdn.b", mixed.value));
        decay = ggml_reshape_4d(g.c(), decay, 1, c.gdn_value_heads, 1, 1);
        beta = ggml_reshape_4d(g.c(), beta, 1, c.gdn_value_heads, 1, 1);
        auto * packed = ggml_gated_delta_net(g.c(), q, k, v, decay, beta, s.recurrent, 1);
        ggml_set_name(packed, (p + "gdn.recurrence").c_str());
        auto * output = ggml_reshape_2d(g.c(), ggml_view_1d(g.c(), packed, vd, 0), c.gdn_dim, c.gdn_value_heads);
        auto * z = ggml_reshape_2d(g.c(), mm(g, p + "gdn.z", mixed.value), c.gdn_dim, c.gdn_value_heads);
        output = ggml_mul(g.c(), norm(g, output, p + "gdn.norm"), ggml_silu(g.c(), z));
        output = mm(g, p + "gdn.out", ggml_reshape_1d(g.c(), output, vd));
        auto * h = c.architecture == Architecture::Qwen35 ? ggml_add(g.c(), hidden, output)
                                                        : combine(g, hidden, output, mixed.inject);
        auto * result = ffn(g, h, p);
        ggml_build_forward_expand(g.graph, result);
        auto * tail = ggml_view_2d(g.c(), history, c.conv_kernel - 1, channels, history->nb[1], sizeof(float));
        g.add(ggml_cpy(g.c(), tail, s.conv));
        auto * next_state = ggml_view_1d(g.c(), packed, int64_t(c.gdn_dim) * vd, size_t(vd) * sizeof(float));
        g.add(ggml_cpy(g.c(), next_state, s.recurrent));
        return result;
    }
    Prepared sparse_prepare(Graph & g, int layer, ggml_tensor * hidden) {
        auto & s = states[layer];
        const std::string p = "blk." + std::to_string(layer) + ".";
        auto mixed = mix(g, hidden, p + "hc_attn");
        auto * qg = mm(g, p + "attn.qg", mixed.value);
        auto * q = ggml_cont(g.c(), ggml_view_2d(g.c(), qg, c.head_dim, c.q_heads,
                                               size_t(2 * c.head_dim) * sizeof(float), 0));
        auto * gate = ggml_cont(g.c(), ggml_view_2d(g.c(), qg, c.head_dim, c.q_heads,
            size_t(2 * c.head_dim) * sizeof(float), size_t(c.head_dim) * sizeof(float)));
        q = rope(g, norm(g, q, p + "attn.q_norm"), c.head_dim, c.q_heads, position);
        auto * k = ggml_reshape_2d(g.c(), mm(g, p + "attn.k", mixed.value), c.head_dim, c.kv_heads);
        k = rope(g, norm(g, k, p + "attn.k_norm"), c.head_dim, c.kv_heads, position);
        auto * v = mm(g, p + "attn.v", mixed.value);
        g.add(q); g.add(k); g.add(v);
        g.add(ggml_cpy(g.c(), gate, s.gate));
        g.add(ggml_cpy(g.c(), mixed.inject, s.inject));

        auto * iq = ggml_reshape_2d(g.c(), mm(g, p + "index.q", mixed.value), c.index_dim, c.index_heads);
        iq = rope(g, norm(g, iq, p + "index.q_norm"), c.index_dim, c.index_heads, position);
        auto * raw = mm(g, p + "index.k", mixed.value);
        const int visible = position + 1, complete = visible / c.block_size;
        if (visible % c.block_size == 0) {
            auto * pooled = raw;
            for (int i = 0; i < c.block_size - 1; ++i)
                pooled = ggml_add(g.c(), pooled, ggml_view_1d(g.c(), s.raw_index, c.index_dim,
                                                           size_t(i) * c.index_dim * sizeof(float)));
            pooled = ggml_scale(g.c(), pooled, 1.0f / c.block_size);
            pooled = rope(g, norm(g, pooled, p + "index.k_norm"), c.index_dim, 1, visible - c.block_size);
            auto * dst = ggml_view_1d(g.c(), s.index_cache, c.index_dim,
                                     size_t(complete - 1) * c.index_dim * sizeof(ggml_bf16_t));
            // This write is explicitly ordered before the scoring graph reads the cache.
            g.add(ggml_cpy(g.c(), pooled, dst));
        }
        auto * raw_slot = ggml_view_1d(g.c(), s.raw_index, c.index_dim,
                                       size_t(position % c.block_size) * c.index_dim * sizeof(float));
        g.add(ggml_cpy(g.c(), raw, raw_slot));
        ggml_tensor * top = nullptr, * scores = nullptr;
        if (complete > 0) {
            auto * cache = ggml_view_2d(g.c(), s.index_cache, c.index_dim, complete, s.index_cache->nb[1], 0);
            // Widen only the compressed index cache, on-device. F32 scoring avoids
            // CPU/CUDA BF16 matvec input-rounding differences at the top-k boundary.
            scores = ggml_mul_mat(g.c(), ggml_cast(g.c(), cache, GGML_TYPE_F32),
                                        ggml_reshape_2d(g.c(), iq, c.index_dim, c.index_heads));
            ggml_prec_set_acc(scores, GGML_PREC_F32);
            ggml_set_name(scores, (p + "index.scores").c_str());
            scores = ggml_relu(g.c(), scores);
            scores = ggml_sum_rows(g.c(), ggml_cont(g.c(), ggml_transpose(g.c(), scores)));
            scores = ggml_scale(g.c(), ggml_reshape_1d(g.c(), scores, complete), 1.0f / std::sqrt(float(c.index_dim)));
            top = ggml_cont(g.c(), ggml_top_k(g.c(), scores, std::min(c.top_blocks, complete)));
            ggml_set_name(top, (p + "index.top_k").c_str());
            g.add(top);
        }
        return {q, k, v, scores, top};
    }
    Selection sparse_layer(int layer, bool capture = true) {
        ProfileScope scope(&profiler, "layer", "qsa", "qsa", position, layer, ggml_backend_name(backend.get()));
        auto & s = states[layer];
        const std::string p = "blk." + std::to_string(layer) + ".";
        auto g = graph("qsa.prepare", layer);
        auto [q, k, v, scores, top] = sparse_prepare(g, layer, hidden);
        const int visible = position + 1;
        if (s.device_keys) return finish_device_attention(g, layer, q, k, v, scores, top, capture);
        g.run();
        auto blocks = top ? g.get<int32_t>(top) : std::vector<int32_t>{};
        ProfileScope expand(&profiler, "cpu", "expand_blocks", "qsa.attention", position, layer, "CPU");
        auto selected = expand_blocks(blocks, visible, c.block_size);
        expand.finish();
        auto queries = g.get<float>(q), keys = g.get<float>(k), values = g.get<float>(v);
        auto start = Clock::now();
        ProfileScope append(&profiler, "cpu", "kv_append_bf16", "qsa.attention", position, layer, "CPU");
        append.event.bytes = (keys.size() + values.size()) * sizeof(ggml_bf16_t);
        s.attention->append(keys, values);
        append.finish();
        ProfileScope attention_scope(&profiler, "cpu", "sparse_attention", "qsa.attention", position, layer, "CPU");
        auto output = s.attention->attend(queries, selected, &profiler, position, layer);
        attention_scope.finish();
        stats.attention_ms += elapsed(start);
        ++stats.sparse_attention_calls;

        auto post = graph("qsa.finish", layer);
        auto * attention = ggml_mul(post.c(), post.floats(output), ggml_sigmoid(post.c(), s.gate));
        auto * h = combine(post, hidden, mm(post, p + "attn.out", attention), s.inject);
        post.add(ggml_cpy(post.c(), ffn(post, h, p), hidden));
        post.run();
        return {layer, std::move(selected)};
    }
    Prepared full_prepare(Graph & g, int layer, ggml_tensor * hidden) {
        auto & s = states[layer];
        const std::string p = "blk." + std::to_string(layer) + ".";
        auto * x = norm(g, hidden, p + "attn_norm");
        auto * qg = mm(g, p + "attn.qg", x);
        auto * q = ggml_cont(g.c(), ggml_view_2d(g.c(), qg, c.head_dim, c.q_heads,
                                               size_t(2*c.head_dim)*sizeof(float), 0));
        auto * gate = ggml_cont(g.c(), ggml_view_2d(g.c(), qg, c.head_dim, c.q_heads,
            size_t(2*c.head_dim)*sizeof(float), size_t(c.head_dim)*sizeof(float)));
        q = rope(g, norm(g, q, p + "attn.q_norm"), c.head_dim, c.q_heads, position);
        auto * k = ggml_reshape_2d(g.c(), mm(g, p + "attn.k", x), c.head_dim, c.kv_heads);
        k = rope(g, norm(g, k, p + "attn.k_norm"), c.head_dim, c.kv_heads, position);
        auto * v = mm(g, p + "attn.v", x);
        g.add(q); g.add(k); g.add(v); g.add(ggml_cpy(g.c(), gate, s.gate));
        return {q, k, v, nullptr, nullptr};
    }
    Selection full_layer(int layer, bool capture = true) {
        ProfileScope scope(&profiler, "layer", "full_attention", "full", position, layer, ggml_backend_name(backend.get()));
        auto & s = states[layer];
        const auto p = "blk." + std::to_string(layer) + ".";
        auto g = graph("full.prepare", layer);
        auto [q, k, v, scores, top] = full_prepare(g, layer, hidden);
        if (s.device_keys) return finish_device_attention(g, layer, q, k, v, nullptr, nullptr, capture);
        g.run();
        auto queries = g.get<float>(q), keys = g.get<float>(k), values = g.get<float>(v);
        std::vector<int32_t> visible(position + 1);
        std::iota(visible.begin(), visible.end(), 0);
        auto start = Clock::now();
        ProfileScope append(&profiler, "cpu", "kv_append_bf16", "full.attention", position, layer, "CPU");
        append.event.bytes = (keys.size() + values.size()) * sizeof(ggml_bf16_t);
        s.attention->append(keys, values);
        append.finish();
        ProfileScope attention_scope(&profiler, "cpu", "full_attention", "full.attention", position, layer, "CPU");
        auto output = s.attention->attend(queries, visible, &profiler, position, layer, "full.attention");
        attention_scope.finish();
        stats.attention_ms += elapsed(start);
        ++stats.full_attention_calls;
        auto post = graph("full.finish", layer);
        auto * attended = ggml_mul(post.c(), post.floats(output), ggml_sigmoid(post.c(), s.gate));
        auto * h = ggml_add(post.c(), hidden, mm(post, p + "attn.out", attended));
        post.add(ggml_cpy(post.c(), ffn(post, h, p), hidden));
        post.run();
        return {layer, std::move(visible)};
    }
    Mixed batch_mix(Graph & g, ggml_tensor * h, const std::string & p, bool inject = true) {
        const int64_t count = h->ne[2];
        auto * flat = ggml_reshape_2d(g.c(), norm(g, h, p + ".norm"), c.hidden * c.branches, count);
        auto * low = ggml_silu(g.c(), ggml_scale(g.c(), mm(g, p + ".down", flat), 1.0f / c.branches));
        auto * gates = ggml_sigmoid(g.c(), mm(g, p + ".up", low));
        auto * gated = ggml_mul(g.c(), flat, gates);
        auto * result = ggml_view_2d(g.c(), gated, c.hidden, count, gated->nb[1], 0);
        for (int branch = 1; branch < c.branches; ++branch)
            result = ggml_add(g.c(), result, ggml_view_2d(g.c(), gated, c.hidden, count,
                                                       gated->nb[1], size_t(branch) * c.hidden * sizeof(float)));
        result = ggml_scale(g.c(), result, 1.0f / c.branches);
        auto * injection = inject ? ggml_scale(g.c(), ggml_sigmoid(g.c(),
            ggml_scale(g.c(), mm(g, p + ".inject", flat), 1.0f / c.branches)), 2.0f) : nullptr;
        return {result, injection};
    }
    ggml_tensor * batch_combine(Graph & g, ggml_tensor * h, ggml_tensor * out, ggml_tensor * injection) {
        const int64_t count = h->ne[2];
        auto * value = ggml_repeat(g.c(), ggml_reshape_3d(g.c(), out, c.hidden, 1, count), h);
        return ggml_add(g.c(), h, ggml_mul(g.c(), value, ggml_reshape_3d(g.c(), injection, 1, c.branches, count)));
    }
    ggml_tensor * token_range(ggml_context * ctx, ggml_tensor * h, int first, int count) {
        return ggml_view_3d(ctx, h, h->ne[0], h->ne[1], count, h->nb[1], h->nb[2], size_t(first)*h->nb[2]);
    }
    template<class F>
    void chunks(const std::string & phase, int layer, int count, uint64_t bytes_per_token,
                int requested, F execute, std::function<void(int, int)> after = {}, bool attention = false, int token_offset = 0) {
        int width = prefill_memory->initial_chunk(count, bytes_per_token, requested);
        if (attention && !requested) width = std::min(width, 4096);
        visit_prefill_chunks(count, width, requested == 0, [&](int first, int n) {
            ProfileScope range(&profiler, "prefill", "chunk", phase, position+token_offset+first, layer, ggml_backend_name(prefill_backend.get()));
            range.event.shape = {n};
            try {
                // execute may only throw PrefillAllocationError before changing
                // model state. Later phases run outside this retry boundary.
                execute(first, n);
            } catch (const PrefillAllocationError & error) {
                range.event.name = "chunk_failed";
                throw PrefillAllocationError(phase + " at token " + std::to_string(position+token_offset+first) +
                    " cannot allocate " + std::to_string(n) + " token(s)" +
                    (requested ? " (explicit size; use auto to reduce it): " : n == 1 ? " (minimum size): " : ": ") + error.what());
            }
        }, [&](int first, int n) {
            ++stats.prefill_chunks;
            auto & minimum = attention ? stats.prefill_attention_min_tile : stats.prefill_chunk_min_tokens;
            auto & maximum = attention ? stats.prefill_attention_max_tile : stats.prefill_chunk_max_tokens;
            minimum = minimum ? std::min<uint64_t>(minimum, n) : n;
            maximum = std::max<uint64_t>(maximum, n);
            if (after) after(first, n);
        }, [&](int first, int n, int smaller) {
            ++stats.prefill_chunk_retries;
            ProfileScope retry(&profiler, "memory", "chunk_reduce", phase, position+token_offset+first, layer, "CPU");
            retry.event.shape = {n, smaller};
        });
    }
    ggml_tensor * batch_moe(Graph & g, ggml_tensor * x, const std::string & p) {
        const int64_t count = x->ne[1];
        auto * probabilities = ggml_soft_max(g.c(), mm(g, p + "router", x));
        auto * ids = ggml_cont(g.c(), ggml_top_k(g.c(), probabilities, c.experts_used));
        ggml_set_name(ids, (p + "expert_ids").c_str());
        auto * selected = ggml_reshape_2d(g.c(), ggml_get_rows(g.c(),
            ggml_reshape_3d(g.c(), probabilities, 1, c.experts, count), ids), c.experts_used, count);
        if (c.norm_topk) selected = ggml_div(g.c(), selected, ggml_sum_rows(g.c(), selected));
        auto expert_mm = [&](const char * name, ggml_tensor * input) {
            auto * out = ggml_mul_mat_id(g.c(), weight(p + name), input, ids);
            ggml_set_name(out, (p + name).c_str()); return out;
        };
        auto * input = ggml_reshape_3d(g.c(), x, c.hidden, 1, count);
        auto * active = ggml_mul(g.c(), ggml_silu(g.c(), expert_mm("gate_exps", input)), expert_mm("up_exps", input));
        auto * outputs = expert_mm("down_exps", active);
        auto * weighted = ggml_mul(g.c(), outputs, ggml_reshape_3d(g.c(), selected, 1, c.experts_used, count));
        auto * result = ggml_view_2d(g.c(), weighted, c.hidden, count, weighted->nb[2], 0);
        for (int i = 1; i < c.experts_used; ++i)
            result = ggml_add(g.c(), result, ggml_view_2d(g.c(), weighted, c.hidden, count,
                                                       weighted->nb[2], size_t(i) * weighted->nb[1]));
        if (c.shared_ffn) {
            auto * shared = ggml_mul(g.c(), ggml_silu(g.c(), mm(g, p + "shared.gate", x)), mm(g, p + "shared.up", x));
            shared = mm(g, p + "shared.down", shared);
            auto * gate = ggml_sigmoid(g.c(), mm(g, p + "shared_gate", x));
            result = ggml_add(g.c(), result, ggml_mul(g.c(), shared, gate));
        }
        return result;
    }
    void batch_ffn(ggml_tensor * batch, int layer) {
        prefill_memory->protect(layer, {model.group(layer, WeightPart::FeedForward)});
        const uint64_t expanded = c.experts ? uint64_t(c.expert_ffn)*c.experts_used+c.shared_ffn : c.ffn;
        chunks("prefill.ffn", layer, int(batch->ne[2]), sizeof(float)*(6*uint64_t(c.hidden)*c.branches+6*expanded),
               options.prefill_chunk_tokens, [&](int first, int count) {
            auto g = graph("prefill.ffn", layer, first);
            auto * h = token_range(g.c(), batch, first, count);
            const auto p = "blk." + std::to_string(layer) + ".";
            ggml_tensor * result;
            if (c.architecture == Architecture::Qwen35) {
                auto * flat = ggml_reshape_2d(g.c(), h, c.hidden, count);
                auto * x = norm(g, flat, p + "ffn_norm");
                auto * active = ggml_mul(g.c(), ggml_silu(g.c(), mm(g, p + "ffn.gate", x)), mm(g, p + "ffn.up", x));
                result = ggml_add(g.c(), flat, mm(g, p + "ffn.down", active));
            } else {
                auto mixed = batch_mix(g, h, p + "hc_ffn");
                auto * output = c.experts ? batch_moe(g, mixed.value, p + "ffn.") :
                    mm(g, p + "ffn.down", ggml_mul(g.c(), ggml_silu(g.c(), mm(g, p + "ffn.gate", mixed.value)),
                                                 mm(g, p + "ffn.up", mixed.value)));
                result = batch_combine(g, h, output, mixed.inject);
            }
            g.add(ggml_cpy(g.c(), result, h));
            if (layer + 1 < c.layers && first+count == batch->ne[2]) g.before_compute = [this, layer] {
                prefill_memory->prefetch({model.group(layer + 1, WeightPart::PLE), model.group(layer + 1, WeightPart::Attention)});
            };
            g.run();
        });
    }
    ggml_tensor * batch_rope(Graph & g, ggml_tensor * x, int dim, int heads, int count, int first) {
        std::vector<int32_t> positions(count); std::iota(positions.begin(), positions.end(), first);
        x = ggml_reshape_3d(g.c(), ggml_cont(g.c(), x), dim, heads, count);
        return ggml_rope_ext(g.c(), x, g.integers(positions), nullptr, c.rope_dim,
                             GGML_ROPE_TYPE_NEOX, c.context, c.rope_base, 1, 0, 1, 0, 0);
    }
    ggml_tensor * batch_gdn_heads(Graph & g, ggml_tensor * x, int count) {
        if (c.architecture == Architecture::Qwen35)
            return ggml_repeat_4d(g.c(), x, c.gdn_dim, c.gdn_value_heads, count, 1);
        ggml_tensor * expanded = nullptr;
        const int repeat = c.gdn_value_heads / c.gdn_key_heads;
        for (int h = 0; h < c.gdn_value_heads; ++h) {
            auto * head = ggml_view_3d(g.c(), x, c.gdn_dim, 1, count, x->nb[1], x->nb[2],
                                      size_t(h / repeat) * x->nb[1]);
            expanded = expanded ? ggml_concat(g.c(), expanded, head, 1) : head;
        }
        return ggml_cont(g.c(), expanded);
    }
    void batch_gdn(ggml_tensor * batch, int layer) {
        prefill_memory->protect(layer, {model.group(layer, WeightPart::Attention)});
        const uint64_t channels = uint64_t(2*c.gdn_key_heads+c.gdn_value_heads)*c.gdn_dim;
        chunks("prefill.gdn", layer, int(batch->ne[2]), sizeof(float)*(6*uint64_t(c.hidden)*c.branches+8*channels),
               options.prefill_chunk_tokens, [&](int first, int count) {
            auto g = graph("prefill.gdn", layer, first);
            auto * h = token_range(g.c(), batch, first, count);
            auto & state = states[layer];
            const auto p = "blk." + std::to_string(layer) + ".";
            auto mixed = c.architecture == Architecture::Qwen35
                ? Mixed{norm(g, ggml_reshape_2d(g.c(), h, c.hidden, count), p + "attn_norm"), nullptr}
                : batch_mix(g, h, p + "hc_attn");
            auto * qkv = mm(g, p + "gdn.qkv", mixed.value);
            const int kd = c.gdn_key_heads * c.gdn_dim, vd = c.gdn_value_heads * c.gdn_dim;
            const int channels = 2 * kd + vd;
            auto * history = ggml_concat(g.c(), state.conv, ggml_cont(g.c(), ggml_transpose(g.c(), qkv)), 0);
            auto * convolved = ggml_silu(g.c(), ggml_ssm_conv(g.c(), history, weight(p + "gdn.conv")));
            auto slice = [&](int heads, int offset) {
                return ggml_cont(g.c(), ggml_view_3d(g.c(), convolved, c.gdn_dim, heads, count,
                    size_t(c.gdn_dim) * sizeof(float), size_t(channels) * sizeof(float), size_t(offset) * sizeof(float)));
            };
            auto * q = slice(c.gdn_key_heads, 0), * k = slice(c.gdn_key_heads, kd);
            auto * v = slice(c.gdn_value_heads, 2 * kd);
            const float scale = 1.0f / std::sqrt(float(c.gdn_dim));
            const float eps = c.architecture == Architecture::Qwen35 ? 1e-6f : c.eps;
            q = batch_gdn_heads(g, ggml_scale(g.c(), ggml_rms_norm(g.c(), q, eps / c.gdn_dim), scale), count);
            k = batch_gdn_heads(g, ggml_scale(g.c(), ggml_rms_norm(g.c(), k, eps / c.gdn_dim), scale), count);
            auto * a = ggml_softplus(g.c(), ggml_add(g.c(), mm(g, p + "gdn.a", mixed.value), weight(p + "gdn.dt_bias")));
            auto * decay = c.architecture == Architecture::Qwen35 ? ggml_mul(g.c(), a, weight(p + "gdn.a_log"))
                : ggml_neg(g.c(), ggml_mul(g.c(), a, ggml_exp(g.c(), weight(p + "gdn.a_log"))));
            auto * beta = ggml_sigmoid(g.c(), mm(g, p + "gdn.b", mixed.value));
            decay = ggml_reshape_4d(g.c(), decay, 1, c.gdn_value_heads, count, 1);
            beta = ggml_reshape_4d(g.c(), beta, 1, c.gdn_value_heads, count, 1);
            auto * packed = ggml_gated_delta_net(g.c(), q, k, v, decay, beta, state.recurrent, 1);
            ggml_set_name(packed, (p + "gdn.batch_recurrence").c_str());
            auto * output = ggml_reshape_3d(g.c(), ggml_view_2d(g.c(), packed, vd, count, size_t(vd) * sizeof(float), 0),
                                           c.gdn_dim, c.gdn_value_heads, count);
            auto * z = ggml_reshape_3d(g.c(), mm(g, p + "gdn.z", mixed.value), c.gdn_dim, c.gdn_value_heads, count);
            output = ggml_mul(g.c(), norm(g, output, p + "gdn.norm"), ggml_silu(g.c(), z));
            output = mm(g, p + "gdn.out", ggml_reshape_2d(g.c(), output, vd, count));
            auto * result = c.architecture == Architecture::Qwen35
                ? ggml_add(g.c(), ggml_reshape_2d(g.c(), h, c.hidden, count), output)
                : batch_combine(g, h, output, mixed.inject);
            g.add(ggml_cpy(g.c(), result, h));
            auto * tail = ggml_view_2d(g.c(), history, c.conv_kernel - 1, channels, history->nb[1], size_t(count) * sizeof(float));
            g.add(ggml_cpy(g.c(), tail, state.conv));
            auto * final_state = ggml_view_1d(g.c(), packed, int64_t(c.gdn_dim) * vd, size_t(vd) * count * sizeof(float));
            g.add(ggml_cpy(g.c(), final_state, state.recurrent));
            g.run();
        });
    }
    void batch_ple(ggml_tensor * batch, int layer, const std::vector<int32_t> & tokens) {
        prefill_memory->protect(layer, {model.group(layer, WeightPart::PLE)});
        chunks("prefill.ple", layer, int(tokens.size()), sizeof(float)*(16*uint64_t(c.hidden)*c.branches+4*uint64_t(c.ple.embedding_dim)),
               options.prefill_chunk_tokens, [&](int first, int count) {
            auto & state = states[layer];
            auto g = graph("prefill.ple", layer, first);
            auto * h = token_range(g.c(), batch, first, count);
            const auto p = "blk." + std::to_string(layer) + ".ple.";
            ProfileScope hash(&profiler, "host", "ngram_hash", "prefill.ple", position+first, layer, "CPU");
            std::vector<int32_t> history_tokens = previous_tokens, rows;
            for (int i = std::max(0, first-c.ple.ngram_size+1); i < first; ++i) {
                history_tokens.push_back(tokens[i]);
                if (history_tokens.size() >= size_t(c.ple.ngram_size)) history_tokens.erase(history_tokens.begin());
            }
            for (int i = first; i < first+count; ++i) {
                const int32_t token = tokens[i];
                auto current = ple_rows(c.ple, *state.ple, token, history_tokens);
                rows.insert(rows.end(), current.begin(), current.end());
                history_tokens.push_back(token);
                if (history_tokens.size() >= size_t(c.ple.ngram_size)) history_tokens.erase(history_tokens.begin());
            }
            hash.finish();
            auto * emb = ggml_reshape_2d(g.c(), ggml_get_rows(g.c(), weight(p + "embedding"), g.integers(rows)), c.ple.embedding_dim, count);
            auto * key = norm(g, ggml_reshape_3d(g.c(), mm(g, p + "key", emb), c.hidden, c.branches, count), p + "norm_key");
            auto * query = norm(g, h, p + "norm_query");
            auto * score = ggml_scale(g.c(), ggml_sum_rows(g.c(), ggml_mul(g.c(), key, query)), 1.0f / std::sqrt(float(c.hidden)));
            auto * magnitude = ggml_sqrt(g.c(), ggml_clamp(g.c(), ggml_abs(g.c(), score), 1e-6f, INFINITY));
            auto * gate = ggml_sigmoid(g.c(), ggml_mul(g.c(), ggml_sgn(g.c(), score), magnitude));
            auto * value = ggml_repeat(g.c(), ggml_reshape_3d(g.c(), mm(g, p + "value", emb), c.hidden, 1, count), h);
            auto * gated = ggml_mul(g.c(), value, gate);
            const int channels = c.hidden * c.branches;
            auto * normalized = ggml_reshape_2d(g.c(), norm(g, gated, p + "norm_conv"), channels, count);
            auto * history = state.ple_conv ? ggml_concat(g.c(), state.ple_conv, normalized, 1) : normalized;
            auto * kernel = weight(p + "conv");
            ggml_tensor * convolved = nullptr;
            for (int k = 0; k < c.ple.conv_kernel; ++k) {
                auto * input = ggml_view_2d(g.c(), history, channels, count, history->nb[1], size_t(k * c.ple.ngram_size) * history->nb[1]);
                auto * tap = ggml_cont(g.c(), ggml_view_2d(g.c(), kernel, 1, channels, kernel->nb[1], k * kernel->nb[0]));
                auto * term = ggml_mul(g.c(), input, ggml_reshape_1d(g.c(), tap, channels));
                convolved = convolved ? ggml_add(g.c(), convolved, term) : term;
            }
            convolved = ggml_reshape_3d(g.c(), ggml_silu(g.c(), convolved), c.hidden, c.branches, count);
            g.add(ggml_cpy(g.c(), ggml_add(g.c(), h, ggml_add(g.c(), gated, convolved)), h));
            if (state.ple_conv) {
                auto * tail = ggml_view_2d(g.c(), history, channels, state.ple_conv->ne[1], history->nb[1], size_t(count) * history->nb[1]);
                g.add(ggml_cpy(g.c(), tail, state.ple_conv));
            }
            g.run();
        });
    }
    void batch_attention(ggml_tensor * batch, int layer, KVExporter * exporter,
                         std::vector<Step> & steps, bool capture) {
        prefill_memory->protect(layer, {model.group(layer, WeightPart::Attention)});
        auto & state = states[layer];
        const int count = int(batch->ne[2]), total = position + count;
        const int dim = attention_dim();
        const int padded = (total + 255) / 256 * 256;
        const int qwidth = c.head_dim * c.q_heads, kvwidth = c.head_dim * c.kv_heads;
        const bool sparse = c.architecture == Architecture::Qwen4Exp;
        const std::string p = "blk." + std::to_string(layer) + ".";
        const bool device_kv = state.device_keys != nullptr;
        std::shared_ptr<PrefillTensorStorage> cache;
        ggml_tensor * keys = nullptr, * values = nullptr;
        PrefillTensorStorage layout(*prefill_memory, prefill_backend.get());
        const auto compute_type = options.prefill_precision == PrefillPrecision::FP16 ? GGML_TYPE_F16 : GGML_TYPE_BF16;
        auto * compute_keys = state.device_keys, * compute_values = state.device_values;
        if (device_kv) {
            if (state.device_kv_size != position) throw std::logic_error("GPU KV prefix is inconsistent");
        } else {
            cache = std::make_shared<PrefillTensorStorage>(*prefill_memory, prefill_backend.get());
            keys = cache->tensor(GGML_TYPE_BF16, kvwidth, total);
            values = cache->tensor(GGML_TYPE_BF16, kvwidth, total);
            cache->allocate();
            compute_keys = layout.tensor(compute_type, dim, padded, c.kv_heads);
            compute_values = layout.tensor(compute_type, dim, padded, c.kv_heads);
            layout.allocate();
        }
        // Only the active layer's KV spans the full context. Projection,
        // permutation and output intermediates are bounded token ranges.
        auto convert = [&](int start, int length) {
            chunks("prefill.attention.layout", layer, length, sizeof(float)*uint64_t(kvwidth)*8,
                   options.prefill_chunk_tokens, [&](int offset, int n) {
                auto g = graph("prefill.attention.layout", layer, start-position+offset);
                auto copy = [&](ggml_tensor * source, ggml_tensor * destination) {
                    auto * range = ggml_view_2d(g.c(), source, kvwidth, n, source->nb[1], size_t(start+offset)*source->nb[1]);
                    auto * shaped = ggml_reshape_3d(g.c(), range, c.head_dim, c.kv_heads, n);
                    auto * wide = ggml_cast(g.c(), shaped, GGML_TYPE_F32);
                    auto * permuted = ggml_cont(g.c(), ggml_permute(g.c(), wide, 0, 2, 1, 3));
                    auto * padded_kv = ggml_pad(g.c(), permuted, dim-c.head_dim, 0, 0, 0);
                    auto * target = ggml_view_3d(g.c(), destination, dim, n, c.kv_heads,
                        destination->nb[1], destination->nb[2], size_t(start+offset)*destination->nb[1]);
                    g.add(ggml_cpy(g.c(), padded_kv, target));
                };
                copy(keys, compute_keys); copy(values, compute_values);
                g.run();
            }, {}, false, start-position);
        };
        if (position && !device_kv) {
            if (state.attention->size() != position) throw std::logic_error("Prefill KV prefix is inconsistent");
            auto g = graph("prefill.attention.prefix", layer);
            const size_t bytes = size_t(position)*kvwidth*sizeof(ggml_bf16_t);
            g.put(keys, state.attention->keys(), bytes);
            g.put(values, state.attention->values(), bytes);
            convert(0, position);
        }
        std::unique_ptr<PrefillTensorStorage> part;
        ggml_tensor * queries = nullptr, * gates = nullptr, * injection = nullptr;
        ggml_tensor * index_queries = nullptr, * output = nullptr;
        chunks("prefill.attention.prepare", layer, count,
               sizeof(float)*(8*uint64_t(c.hidden)*c.branches+12*uint64_t(qwidth)+8*uint64_t(kvwidth)),
               options.prefill_chunk_tokens, [&](int first, int n) {
            // Allocation and this producer graph are one retry boundary. No
            // later attention/layout failure can replay its QSA state updates.
            part.reset();
            part = std::make_unique<PrefillTensorStorage>(*prefill_memory, prefill_backend.get());
            queries = part->tensor(GGML_TYPE_F32, c.head_dim, c.q_heads, n);
            gates = part->tensor(GGML_TYPE_F32, qwidth, n);
            injection = sparse ? part->tensor(GGML_TYPE_F32, c.branches, n) : nullptr;
            index_queries = sparse ? part->tensor(GGML_TYPE_F32, c.index_dim, c.index_heads, n) : nullptr;
            output = part->tensor(GGML_TYPE_F32, qwidth, n);
            part->allocate();
            auto g = graph("prefill.attention.prepare", layer, first);
            auto * h = token_range(g.c(), batch, first, n);
            const int start = position+first;
            auto mixed = sparse ? batch_mix(g, h, p+"hc_attn")
                : Mixed{norm(g, ggml_reshape_2d(g.c(), h, c.hidden, n), p+"attn_norm"), nullptr};
            auto * qg = mm(g, p+"attn.qg", mixed.value);
            auto * q = ggml_cont(g.c(), ggml_view_3d(g.c(), qg, c.head_dim, c.q_heads, n,
                size_t(2*c.head_dim)*sizeof(float), qg->nb[1], 0));
            auto * gate = ggml_cont(g.c(), ggml_view_3d(g.c(), qg, c.head_dim, c.q_heads, n,
                size_t(2*c.head_dim)*sizeof(float), qg->nb[1], size_t(c.head_dim)*sizeof(float)));
            q = batch_rope(g, norm(g, q, p+"attn.q_norm"), c.head_dim, c.q_heads, n, start);
            auto * k = ggml_reshape_3d(g.c(), mm(g, p+"attn.k", mixed.value), c.head_dim, c.kv_heads, n);
            k = batch_rope(g, norm(g, k, p+"attn.k_norm"), c.head_dim, c.kv_heads, n, start);
            auto * v = mm(g, p+"attn.v", mixed.value);
            g.add(ggml_cpy(g.c(), q, queries));
            g.add(ggml_cpy(g.c(), gate, gates));
            auto suffix = [&](ggml_tensor * t) {
                return ggml_view_2d(g.c(), t, kvwidth, n, t->nb[1], size_t(start)*t->nb[1]);
            };
            if (device_kv) append_device_kv(g, state, k, v, start, n);
            else {
                g.add(ggml_cpy(g.c(), k, suffix(keys)));
                g.add(ggml_cpy(g.c(), v, suffix(values)));
            }
            if (sparse) {
                g.add(ggml_cpy(g.c(), mixed.inject, injection));
                auto * iq = ggml_reshape_3d(g.c(), mm(g, p+"index.q", mixed.value), c.index_dim, c.index_heads, n);
                iq = batch_rope(g, norm(g, iq, p+"index.q_norm"), c.index_dim, c.index_heads, n, start);
                g.add(ggml_cpy(g.c(), iq, index_queries));
                auto * raw = mm(g, p+"index.k", mixed.value);
                const int old_tail = start%c.block_size;
                if (old_tail) raw = ggml_concat(g.c(), ggml_view_2d(g.c(), state.raw_index,
                    c.index_dim, old_tail, state.raw_index->nb[1], 0), raw, 1);
                const int new_blocks = (old_tail+n)/c.block_size;
                if (new_blocks) {
                    auto * complete = ggml_cont(g.c(), ggml_view_2d(g.c(), raw, c.index_dim,
                        new_blocks*c.block_size, raw->nb[1], 0));
                    complete = ggml_reshape_3d(g.c(), complete, c.index_dim, c.block_size, new_blocks);
                    auto * pooled = ggml_reshape_3d(g.c(), ggml_mean(g.c(), ggml_cont(g.c(),
                        ggml_permute(g.c(), complete, 1, 0, 2, 3))), c.index_dim, 1, new_blocks);
                    pooled = norm(g, pooled, p+"index.k_norm");
                    std::vector<int32_t> starts(new_blocks);
                    for (int j=0; j<new_blocks; ++j) starts[j] = (start/c.block_size+j)*c.block_size;
                    pooled = ggml_rope_ext(g.c(), pooled, g.integers(starts), nullptr, c.rope_dim,
                        GGML_ROPE_TYPE_NEOX, c.context, c.rope_base, 1, 0, 1, 0, 0);
                    auto * destination = ggml_view_2d(g.c(), state.index_cache, c.index_dim, new_blocks,
                        state.index_cache->nb[1], size_t(start/c.block_size)*state.index_cache->nb[1]);
                    g.add(ggml_cpy(g.c(), pooled, destination));
                }
                const int tail = (start+n)%c.block_size;
                if (tail) {
                    auto * source = ggml_view_2d(g.c(), raw, c.index_dim, tail, raw->nb[1], size_t(new_blocks*c.block_size)*raw->nb[1]);
                    auto * destination = ggml_view_2d(g.c(), state.raw_index, c.index_dim, tail, state.raw_index->nb[1], 0);
                    g.add(ggml_cpy(g.c(), source, destination));
                }
            }
            g.run();
            if (device_kv) state.device_kv_size = start+n;
        }, [&](int first, int n) {
            if (!device_kv) convert(position+first, n);
            const int visible_end = position+first+n;
            const int visible_padded = (visible_end+255)/256*256;
            const int blocks = visible_end/c.block_size;
            const uint64_t attention_bytes = sizeof(float)*(4*uint64_t(visible_padded)+
                (sparse ? 4*uint64_t(blocks)*c.index_heads : 0)+8*uint64_t(qwidth));
            chunks("prefill.attention", layer, n, attention_bytes, options.prefill_attention_tile,
                   [&](int offset, int queries_count) {
                auto g = graph("prefill.attention", layer, first+offset);
                auto * q = ggml_view_3d(g.c(), queries, c.head_dim, c.q_heads, queries_count,
                    queries->nb[1], queries->nb[2], size_t(offset)*queries->nb[2]);
                auto * mask = ggml_fill(g.c(), ggml_new_tensor_2d(g.c(), GGML_TYPE_F32, visible_padded, queries_count), 0);
                mask = ggml_diag_mask_inf(g.c(), mask, position+first+offset);
                ggml_tensor * top = nullptr;
                if (sparse && blocks) {
                    auto * iq = ggml_cont(g.c(), ggml_view_3d(g.c(), index_queries, c.index_dim, c.index_heads, queries_count,
                        index_queries->nb[1], index_queries->nb[2], size_t(offset)*index_queries->nb[2]));
                    auto * compressed = ggml_view_2d(g.c(), state.index_cache, c.index_dim, blocks, state.index_cache->nb[1], 0);
                    auto * scores = ggml_mul_mat(g.c(), ggml_cast(g.c(), compressed, GGML_TYPE_F32),
                        ggml_reshape_2d(g.c(), iq, c.index_dim, c.index_heads*queries_count));
                    ggml_prec_set_acc(scores, GGML_PREC_F32);
                    scores = ggml_reshape_3d(g.c(), ggml_relu(g.c(), scores), blocks, c.index_heads, queries_count);
                    scores = ggml_reshape_2d(g.c(), ggml_sum_rows(g.c(), ggml_cont(g.c(), ggml_permute(g.c(), scores, 1, 0, 2, 3))), blocks, queries_count);
                    scores = ggml_scale(g.c(), scores, 1.0f/std::sqrt(float(c.index_dim)));
                    std::vector<float> completed(queries_count);
                    std::vector<int32_t> extra(size_t(queries_count)*c.block_size, -1), mapping(size_t(blocks)*c.block_size);
                    std::iota(mapping.begin(), mapping.end(), 0);
                    for (int i=0; i<queries_count; ++i) {
                        const int visible = position+first+offset+i+1, full = visible/c.block_size;
                        completed[i] = float(full);
                        for (int j=full*c.block_size; j<visible; ++j) extra[size_t(i)*c.block_size+j-full*c.block_size] = j;
                    }
                    auto * eligible = ggml_sub(g.c(), ggml_repeat(g.c(), ggml_reshape_2d(g.c(), g.floats(completed), 1, queries_count), scores),
                        ggml_arange(g.c(), 0, float(blocks), 1));
                    scores = ggml_add(g.c(), scores, ggml_log(g.c(), ggml_step(g.c(), eligible)));
                    top = ggml_cont(g.c(), ggml_top_k(g.c(), scores, std::min(c.top_blocks, blocks)));
                    ggml_set_name(top, (p+"index.batch_top_k").c_str());
                    mask = ggml_top_k_mask(g.c(), mask, top, scores,
                        ggml_reshape_2d(g.c(), g.integers(mapping), c.block_size, blocks),
                        ggml_reshape_2d(g.c(), g.integers(extra), c.block_size, queries_count));
                    if (capture) g.add(top);
                }
                auto * cropped = flash_attention(g, q, compute_keys, compute_values, mask, queries_count, visible_end);
                auto * target = ggml_view_2d(g.c(), output, qwidth, queries_count, output->nb[1], size_t(offset)*output->nb[1]);
                g.add(ggml_cpy(g.c(), cropped, target));
                g.run(); ++stats.prefill_gpu_attention_calls;
                if (capture) {
                    const auto ids = top ? g.get<int32_t>(top) : std::vector<int32_t>{};
                    for (int i=0; i<queries_count; ++i) {
                        const int visible = position+first+offset+i+1;
                        std::vector<int32_t> selected;
                        if (sparse) {
                            std::vector<int32_t> chosen;
                            if (top) for (int j=0; j<top->ne[0]; ++j) {
                                const int id = ids[size_t(i)*top->ne[0]+j];
                                if (id < visible/c.block_size) chosen.push_back(id);
                            }
                            selected = expand_blocks(chosen, visible, c.block_size);
                        } else { selected.resize(visible); std::iota(selected.begin(), selected.end(), 0); }
                        steps[first+offset+i].selections.push_back({layer, std::move(selected)});
                    }
                }
            }, {}, true, first);
            chunks("prefill.attention.finish", layer, n, sizeof(float)*(6*uint64_t(c.hidden)*c.branches+4*uint64_t(qwidth)),
                   options.prefill_chunk_tokens, [&](int offset, int size) {
                auto g = graph("prefill.attention.finish", layer, first+offset);
                auto * h = token_range(g.c(), batch, first+offset, size);
                auto range = [&](ggml_tensor * t) {
                    return ggml_view_2d(g.c(), t, t->ne[0], size, t->nb[1], size_t(offset)*t->nb[1]);
                };
                auto * attended = ggml_mul(g.c(), range(output), ggml_sigmoid(g.c(), range(gates)));
                auto * projected = mm(g, p+"attn.out", attended);
                auto * result = sparse ? batch_combine(g, h, projected, range(injection))
                    : ggml_add(g.c(), ggml_reshape_2d(g.c(), h, c.hidden, size), projected);
                g.add(ggml_cpy(g.c(), result, h));
                if (first+offset+size == count)
                    g.before_compute = [this, layer] { prefill_memory->prefetch({model.group(layer, WeightPart::FeedForward)}); };
                g.run();
            }, {}, false, first);
            part.reset();
        });
        // The complete layer is now immutable. Export once, retaining the source
        // until the separate stream finishes; the next layer still starts later.
        if (exporter) exporter->enqueue(cache, keys, values, *state.attention, position, count, layer);
    }

    std::vector<Step> prefill(const std::vector<int32_t> & tokens, bool capture) {
        if (failed) throw std::runtime_error("Previous execution failed; call reset() before prefill");
        decode_graph.reset();
        if (tokens.size() > size_t(c.context-position)) throw std::out_of_range("Prompt exceeds context capacity");
        for (auto token : tokens) if (token < 0 || token >= c.vocab) throw std::out_of_range("Prompt token outside vocabulary");
        if (tokens.empty()) return {};
        std::vector<Step> result; result.reserve(tokens.size());
        if (options.backend == "cpu") {
            for (auto token : tokens) result.push_back(decode(token, capture));
            return result;
        }
        PrefillMemory memory(model, backend.get(), stats, profiler, options);
        prefill_memory = &memory;
        try {
            prefill_backend = make_backend(options);
            ProfileScope scope(&profiler, "prefill", "layer_major", "prefill", position, -1, ggml_backend_name(prefill_backend.get()));
            {
                std::unique_ptr<KVExporter> exporter;
                if (options.kv_placement == KVPlacement::CPU) {
                    exporter = std::make_unique<KVExporter>(options.device, stats, profiler);
                    memory.reclaim_completed = [&] { exporter->reap(); };
                    memory.wait_transfers = [&] { exporter->finish(); };
                }
                const int count = int(tokens.size());
                for (int i=0; i<count; ++i) result.push_back({position+i, {}, {}});
                const int embedding_group = c.tied_embeddings ? model.group(c.layers, WeightPart::Head) : model.group(-1, WeightPart::Embedding);
                memory.protect(-1, {embedding_group});
                PrefillTensorStorage activations(memory, prefill_backend.get());
                auto * batch = activations.tensor(GGML_TYPE_F32, c.hidden, c.branches, count);
                auto * finite_flag = activations.tensor(GGML_TYPE_F32, 1);
                activations.allocate();
                chunks("prefill.embedding", -1, count, sizeof(float)*uint64_t(c.hidden)*(2*c.branches+1),
                       options.prefill_chunk_tokens, [&](int first, int n) {
                    auto g = graph("prefill.embedding", -1, first);
                    auto * target = token_range(g.c(), batch, first, n);
                    auto * embedded = ggml_get_rows(g.c(), weight("token_embd.weight"),
                                                   g.integers({tokens.begin()+first, tokens.begin()+first+n}));
                    embedded = ggml_reshape_3d(g.c(), embedded, c.hidden, 1, n);
                    g.add(ggml_cpy(g.c(), ggml_repeat(g.c(), embedded, target), target)); g.run();
                });
                for (int layer=0; layer<c.layers; ++layer) {
                    ProfileScope layer_scope(&profiler, "layer", "batch", "prefill.layer", position, layer, ggml_backend_name(prefill_backend.get()));
                    if (states[layer].ple) batch_ple(batch, layer, tokens);
                    if (c.is_recurrent(layer)) batch_gdn(batch, layer);
                    else batch_attention(batch, layer, exporter.get(), result, capture);
                    batch_ffn(batch, layer);
                    ++stats.prefill_layers;
                }
                memory.protect(c.layers, {model.group(c.layers, WeightPart::Head)});
                auto project = [&](Graph & g, ggml_tensor * source) {
                        auto mixed = c.architecture == Architecture::Qwen35
                            ? Mixed{norm(g, ggml_reshape_2d(g.c(), source, c.hidden, source->ne[2]), "output_norm.weight"), nullptr}
                            : batch_mix(g, source, "head", false);
                        return mm(g, "output.weight", mixed.value);
                };
                chunks("prefill.head", c.layers, count,
                       sizeof(float)*(6*uint64_t(c.hidden)*c.branches+(capture ? 2*uint64_t(c.vocab) : 0)),
                       options.prefill_chunk_tokens, [&](int first, int n) {
                    auto g = graph("prefill.head", c.layers, first);
                    auto * source = token_range(g.c(), batch, first, n);
                    auto * logits = capture ? project(g, source) : nullptr;
                    if (logits) g.add(logits);
                    auto * finite = ggml_sum(g.c(), ggml_sub(g.c(), source, source));
                    if (logits) finite = ggml_add(g.c(), finite, ggml_sum(g.c(), ggml_sub(g.c(), logits, logits)));
                    g.add(ggml_cpy(g.c(), ggml_add(g.c(), finite_flag, finite), finite_flag));
                    g.run();
                    if (capture) {
                        auto all = g.get<float>(logits);
                        for (int i=0; i<n; ++i) result[first+i].logits.assign(all.begin()+size_t(i)*c.vocab, all.begin()+size_t(i+1)*c.vocab);
                    }
                });
                {
                    // Preserve the same final GEMV arithmetic with and without
                    // diagnostics, independent of the intermediate chunk sizes.
                    auto g = graph("prefill.head", c.layers, count-1);
                    auto * tail = token_range(g.c(), batch, count-1, 1);
                    auto * last = project(g, tail);
                    auto * finite = ggml_add(g.c(), finite_flag, ggml_sum(g.c(), ggml_sub(g.c(), last, last)));
                    g.add(finite); g.add(ggml_cpy(g.c(), last, last_logits));
                    g.add(ggml_cpy(g.c(), tail, hidden)); g.run();
                    if (!std::isfinite(g.get<float>(finite)[0])) throw std::runtime_error("Non-finite logits or hidden values during prefill");
                    if (capture) result.back().logits = g.get<float>(last_logits);
                }
                if (exporter) exporter->finish();
            }
            memory.reclaim_completed = {}; memory.wait_transfers = {};
            // Release kernel-pool scratch before restoring the decode weight layout.
            prefill_backend.reset();
            memory.restore();
            prefill_memory = nullptr;
            captured_logits = capture ? result.back().logits : std::vector<float>{};
            for (auto token : tokens) {
                if (!c.ple.layers.empty()) {
                    previous_tokens.push_back(token);
                    if (previous_tokens.size() >= size_t(c.ple.ngram_size)) previous_tokens.erase(previous_tokens.begin());
                }
                sampler->accept(token);
            }
            position += int(tokens.size()); stats.prefill_tokens += tokens.size();
            return result;
        } catch (...) {
            memory.reclaim_completed = {}; memory.wait_transfers = {};
            prefill_backend.reset(); prefill_memory = nullptr; failed = true;
            throw;
        }
    }
    // One graph from embedding to logits. Builders chain activations instead of
    // round-tripping through the persistent hidden tensor between layers.
    void decode_single_graph(int32_t token) {
        const int padded = (position+256)/256*256;
        // Qwen3.5 topology depends only on the 256-token KV bucket; QSA and PLE
        // graphs carry position-dependent structure and are rebuilt per token.
        const bool reusable = c.architecture == Architecture::Qwen35 && c.ple.layers.empty();
        const bool rebuild = !decode_graph || !reusable || decode_graph->kv_padded != padded;
        if (rebuild) {
            decode_graph.reset();
            decode_graph = std::make_unique<Graph>(backend.get(), stats, options.backend == "cuda", profiler, "decode", position, -1,
                                                   nullptr, decode_allocator_get(), 16384);
            auto & g = *decode_graph;
            g.kv_padded = padded;
            g.token_input = g.input(GGML_TYPE_I32, 1);
            g.position_input = g.input(GGML_TYPE_I32, 1);
            g.kv_row_input = g.input(GGML_TYPE_I64, 1);
            // Qwen3.5 consumes the mask directly as F16; QSA derives its sparse
            // mask from an F32 causal mask.
            g.mask_input = g.input(reusable ? GGML_TYPE_F16 : GGML_TYPE_F32, padded);
            // Only one mask entry is updated per token, so the allocator must not
            // reuse this tensor's memory for intermediates after its last reader.
            ggml_set_output(g.mask_input);
            auto * embedded = ggml_get_rows(g.c(), weight("token_embd.weight"), g.token_input);
            auto * h = c.branches == 1 ? ggml_reshape_2d(g.c(), embedded, c.hidden, 1) : ggml_repeat(g.c(), embedded, hidden);
            for (int layer = 0; layer < c.layers; ++layer) {
                if (states[layer].ple) h = build_ple(g, layer, token, h);
                if (c.is_recurrent(layer)) { h = build_linear(g, layer, h); continue; }
                auto prepared = c.architecture == Architecture::Qwen35 ? full_prepare(g, layer, h) : sparse_prepare(g, layer, h);
                h = device_attention(g, layer, h, prepared.q, prepared.k, prepared.v, prepared.scores, prepared.top);
            }
            auto mixed = c.architecture == Architecture::Qwen35
                ? Mixed{norm(g, h, "output_norm.weight"), nullptr} : mix(g, h, "head", false);
            auto * logits = mm(g, "output.weight", mixed.value);
            g.add(ggml_cpy(g.c(), logits, last_logits));
            g.add(ggml_cpy(g.c(), h, hidden));
            g.finite = ggml_sum(g.c(), ggml_sub(g.c(), logits, logits));
            g.output(g.finite);
            g.reserve();
            std::vector<float> mask(padded, -INFINITY);
            std::fill(mask.begin(), mask.begin()+position+1, 0.0f);
            if (g.mask_input->type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> half(mask.size());
                ggml_fp32_to_fp16_row(mask.data(), half.data(), int64_t(mask.size()));
                g.put(g.mask_input, half.data(), half.size()*sizeof(ggml_fp16_t));
            } else {
                g.put(g.mask_input, mask.data(), mask.size()*sizeof(float));
            }
        } else {
            const ggml_fp16_t unmasked = ggml_fp32_to_fp16(0.0f);
            decode_graph->put_at(decode_graph->mask_input, &unmasked, size_t(position)*sizeof(unmasked), sizeof(unmasked));
        }
        auto & g = *decode_graph;
        const int64_t row = position;
        g.put(g.token_input, &token, sizeof(token));
        g.put(g.position_input, &position, sizeof(position));
        g.put(g.kv_row_input, &row, sizeof(row));
        g.run();
        if (!rebuild) ++stats.decode_graph_reuses;
        stats.gpu_attention_calls += rebuild ? 0 : uint64_t(std::count_if(states.begin(), states.end(),
            [](const LayerState & s) { return s.device_keys != nullptr; }));
        for (int layer = 0; layer < c.layers; ++layer)
            if (states[layer].device_keys) states[layer].device_kv_size = position + 1;
        if (!std::isfinite(g.get<float>(g.finite)[0])) throw std::runtime_error("Non-finite logits");
        if (!reusable) decode_graph.reset();
    }
    Step decode(int32_t token, bool capture_logits) {
        if (failed) throw std::runtime_error("Previous decode failed; call reset() before reusing the engine");
        if (token < 0 || token >= c.vocab) throw std::out_of_range("Token ID outside vocabulary");
        if (position >= c.context) throw std::out_of_range("Context capacity exceeded; reset before decoding a new sequence");
        try {
            ProfileScope token_scope(&profiler, "token", "decode", "decode", position, -1, "CPU");
            if (single_graph_decode(capture_logits)) {
                Step step{position, {}, {}};
                decode_single_graph(token);
                captured_logits.clear();
                if (!c.ple.layers.empty()) {
                    previous_tokens.push_back(token);
                    if (previous_tokens.size() >= size_t(c.ple.ngram_size)) previous_tokens.erase(previous_tokens.begin());
                }
                sampler->accept(token);
                ++position;
                return step;
            }
            auto emb = graph("embedding");
            auto * token_embedding = ggml_get_rows(emb.c(), weight("token_embd.weight"), emb.integer(token));
            emb.add(ggml_cpy(emb.c(), ggml_repeat(emb.c(), token_embedding, hidden), hidden));
            emb.run();
            Step step{position, {}, {}};
            for (int layer = 0; layer < c.layers; ++layer) {
                if (states[layer].ple) ple_layer(layer, token);
                if (c.is_recurrent(layer)) linear_layer(layer);
                else {
                    auto selection = c.architecture == Architecture::Qwen35 ? full_layer(layer, capture_logits) : sparse_layer(layer, capture_logits);
                    if (capture_logits || options.kv_placement == KVPlacement::CPU)
                        step.selections.push_back(std::move(selection));
                }
            }
            auto head = graph("head");
            auto mixed = c.architecture == Architecture::Qwen35
                ? Mixed{norm(head, hidden, "output_norm.weight"), nullptr} : mix(head, hidden, "head", false);
            auto * logits = mm(head, "output.weight", mixed.value);
            head.add(ggml_cpy(head.c(), logits, last_logits));
            // x-x is zero for finite x, NaN for NaN/Inf. Validate every prompt
            // position without downloading its full vocabulary vector.
            auto * finite = capture_logits ? nullptr : ggml_sum(head.c(), ggml_sub(head.c(), logits, logits));
            if (finite) head.add(finite);
            head.run();
            captured_logits.clear();
            if (capture_logits) step.logits = head.get<float>(last_logits);
            if ((finite && !std::isfinite(head.get<float>(finite)[0])) ||
                !std::all_of(step.logits.begin(), step.logits.end(), [](float f) { return std::isfinite(f); }))
                throw std::runtime_error("Non-finite logits");
            captured_logits = step.logits;
            if (!c.ple.layers.empty()) {
                previous_tokens.push_back(token);
                if (previous_tokens.size() >= size_t(c.ple.ngram_size)) previous_tokens.erase(previous_tokens.begin());
            }
            sampler->accept(token);
            ++position;
            return step;
        } catch (...) { failed = true; throw; }
    }
    int32_t sample() {
        if (failed) throw std::runtime_error("Previous execution failed; call reset() before sampling");
        if (!position) throw std::logic_error("Decode at least one token before sampling");
        try { return sampler->sample(last_logits, captured_logits.empty() ? nullptr : &captured_logits, position - 1); }
        catch (...) { failed = true; throw; }
    }
    void reset() {
        decode_graph.reset();
        model.wait_all();
        for (size_t id=0; id<model.groups.size(); ++id) model.load(int(id));
        ggml_backend_synchronize(backend.get());
        ggml_backend_buffer_clear(state_buffer.get(), 0);
        for (auto & s : states) {
            if (s.attention) s.attention->reset();
            s.device_kv_size = 0;
        }
        if (kv_token_ids) {
            std::vector<int32_t> ids(c.context); std::iota(ids.begin(), ids.end(), 0);
            ggml_backend_tensor_set(kv_token_ids, ids.data(), 0, ids.size()*sizeof(int32_t));
        }
        previous_tokens.clear();
        captured_logits.clear();
        sampler->reset();
        position = 0;
        failed = false;
        stats = {};
        stats.gpu_kv_bytes = gpu_kv_bytes;
        profiler.clear();
    }
};

Engine::Engine(const std::string & path, Options options) : impl_(std::make_unique<Impl>(path, std::move(options))) {}
Engine::~Engine() = default;
Step Engine::decode(int32_t token, bool capture_logits) { return impl_->decode(token, capture_logits); }
int32_t Engine::sample() { return impl_->sample(); }
std::string Engine::sampling_backend_name() const { return impl_->sampler->backend_name(); }
std::vector<Step> Engine::prefill(const std::vector<int32_t> & tokens, bool capture_logits) {
    return impl_->prefill(tokens, capture_logits);
}
void Engine::reset() { impl_->reset(); }
const Config & Engine::config() const { return impl_->c; }
const Statistics & Engine::statistics() const { return impl_->stats; }
const ModelInfo & Engine::model_info() const { return impl_->model.info; }
std::string Engine::backend_name() const { return ggml_backend_name(impl_->backend.get()); }
std::string Engine::backend_description() const {
    return ggml_backend_dev_description(ggml_backend_get_device(impl_->backend.get()));
}
void Engine::set_profile(ProfileMode mode) { impl_->profiler.mode = mode; impl_->profiler.clear(); }
const std::vector<ProfileEvent> & Engine::profile_events() const { return impl_->profiler.events; }
uint64_t Engine::profile_dropped_events() const { return impl_->profiler.dropped; }
} // namespace tidal
