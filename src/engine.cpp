#include "internal.hpp"
#include "sampling.hpp"
#include "prefill_tensor.hpp"
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
    Context ctx = context(4 * 1024 * 1024);
    Buffer storage{nullptr, ggml_backend_buffer_free};
    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 2048, false);
    ggml_backend_t backend;
    PrefillMemory * memory = nullptr;
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator{nullptr, ggml_gallocr_free};
    size_t allocated_bytes = 0;
    std::function<void()> before_compute;
    Statistics & stats;
    bool device;
    std::vector<std::function<void()>> uploads;
    Graph(ggml_backend_t b, Statistics & s, bool d, Profiler & p,
          std::string ph, int pos, int l, PrefillMemory * mem = nullptr)
        : profiler(p), phase(std::move(ph)), backend_label(ggml_backend_name(b)), position(pos), layer(l),
          build_scope(&p, "host", "graph_build", phase, pos, l, "CPU"),
          backend(b), memory(mem), stats(s), device(d) {}
    ~Graph() {
        ProfileScope release(&profiler, "host", "graph_free", phase, position, layer, "CPU");
        storage.reset();
        allocator.reset();
        if (memory && allocated_bytes) memory->release(allocated_bytes);
        ctx.reset();
    }
    ggml_context * c() { return ctx.get(); }
    void add(ggml_tensor * t) { if (memory) ggml_set_output(t); ggml_build_forward_expand(graph, t); }
    ggml_tensor * integer(int32_t value) {
        return integers({value});
    }
    ggml_tensor * integers(const std::vector<int32_t> & values) {
        auto * t = ggml_new_tensor_1d(c(), GGML_TYPE_I32, int64_t(values.size()));
        if (memory) ggml_set_input(t);
        uploads.emplace_back([this, t, values] { put(t, values.data(), values.size() * sizeof(int32_t)); });
        return t;
    }
    ggml_tensor * floats(const std::vector<float> & values) {
        auto * t = ggml_new_tensor_1d(c(), GGML_TYPE_F32, int64_t(values.size()));
        if (memory) ggml_set_input(t);
        uploads.emplace_back([this, t, values] { put(t, values.data(), values.size() * sizeof(float)); });
        return t;
    }
    void put(ggml_tensor * t, const void * data, size_t size) {
        ProfileScope transfer(&profiler, "transfer", device ? "H2D" : "host_write", phase, position, layer, backend_label);
        transfer.event.bytes = size;
        auto start = Clock::now();
        ggml_backend_tensor_set(t, data, 0, size);
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
    void run() {
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
            if (!ggml_gallocr_reserve(allocator.get(), graph)) throw std::runtime_error("Prefill graph reservation failed");
            if (!ggml_gallocr_alloc_graph(allocator.get(), graph)) throw std::runtime_error("Prefill graph allocation failed");
        } else {
            storage.reset(ggml_backend_alloc_ctx_tensors(c(), backend));
            if (!storage) throw std::runtime_error("Graph allocation failed");
        }
        allocation.finish();
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
};
struct Mixed { ggml_tensor * value; ggml_tensor * inject; };
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
    std::vector<float> captured_logits;
    std::unique_ptr<Sampler> sampler;
    std::vector<LayerState> states;
    std::vector<int32_t> previous_tokens;
    int position = 0;
    bool failed = false;
    PrefillMemory * prefill_memory = nullptr;
    Backend prefill_backend{nullptr, ggml_backend_free};

    Impl(const std::string & path, Options opts)
        : options(std::move(opts)), backend(make_backend(options)), model(path, backend.get()), c(model.config),
          profiler(options.profile, options.nvtx, options.profile_event_limit), state_context(context(1024 * 1024)), states(c.layers) {
        if (options.prefill_attention_tile < 1 || options.prefill_attention_tile > 4096 ||
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
                s.attention = std::make_unique<SparseAttention>(c.q_heads, c.kv_heads, c.head_dim,
                                                               c.context, options.threads);
            }
        }
        for (const auto & l : c.ple.layers) {
            auto & s = states[l.layer];
            s.ple = &l;
            if (c.ple.conv_kernel > 1)
                s.ple_conv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.hidden * c.branches,
                                               (c.ple.conv_kernel - 1) * c.ple.ngram_size);
        }
        state_buffer.reset(ggml_backend_alloc_ctx_tensors(ctx, backend.get()));
        if (!state_buffer) throw std::runtime_error("State allocation failed");
        sampler = std::make_unique<Sampler>(c.vocab, options.sampling, backend.get(), options.device, stats, profiler);
        reset();
    }
    Graph graph(const std::string & phase, int layer = -1, int token_offset = 0) {
        return Graph(prefill_backend ? prefill_backend.get() : backend.get(), stats, options.backend == "cuda", profiler, phase, position + token_offset, layer, prefill_memory);
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
        x = ggml_reshape_3d(g.c(), ggml_cont(g.c(), x), dim, heads, 1);
        return ggml_rope_ext(g.c(), x, g.integer(pos), nullptr, c.rope_dim,
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
            auto * active = ggml_mul(g.c(), ggml_silu(g.c(), mm(g, p + "ffn.gate", x)), mm(g, p + "ffn.up", x));
            return ggml_add(g.c(), h, mm(g, p + "ffn.down", active));
        }
        auto mixed = mix(g, h, p + "hc_ffn");
        if (c.experts) return combine(g, h, moe(g, mixed.value, p + "ffn."), mixed.inject);
        auto * v = ggml_mul(g.c(), ggml_silu(g.c(), mm(g, p + "ffn.gate", mixed.value)),
                           mm(g, p + "ffn.up", mixed.value));
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
        auto & s = states[layer];
        ProfileScope scope(&profiler, "layer", "ple", "ple", position, layer, ggml_backend_name(backend.get()));
        ProfileScope hash(&profiler, "host", "ngram_hash", "ple", position, layer, "CPU");
        auto rows = ple_rows(c.ple, *s.ple, token, previous_tokens);
        hash.finish();
        auto g = graph("ple", layer);
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
        g.add(ggml_cpy(g.c(), ggml_add(g.c(), hidden, ggml_add(g.c(), gated, convolved)), hidden));
        if (s.ple_conv) {
            auto * tail = ggml_view_2d(g.c(), history, channels, s.ple_conv->ne[1], history->nb[1], history->nb[1]);
            g.add(ggml_cpy(g.c(), tail, s.ple_conv));
        }
        g.run();
    }
    void linear_layer(int layer) {
        ProfileScope scope(&profiler, "layer", "gdn", "gdn", position, layer, ggml_backend_name(backend.get()));
        auto g = graph("gdn", layer);
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
        q = repeat_gdn_heads(g, ggml_scale(g.c(), ggml_rms_norm(g.c(), q, l2_eps / c.gdn_dim), scale));
        k = repeat_gdn_heads(g, ggml_scale(g.c(), ggml_rms_norm(g.c(), k, l2_eps / c.gdn_dim), scale));
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
        g.add(ggml_cpy(g.c(), ffn(g, h, p), hidden));
        auto * tail = ggml_view_2d(g.c(), history, c.conv_kernel - 1, channels, history->nb[1], sizeof(float));
        g.add(ggml_cpy(g.c(), tail, s.conv));
        auto * next_state = ggml_view_1d(g.c(), packed, int64_t(c.gdn_dim) * vd, size_t(vd) * sizeof(float));
        g.add(ggml_cpy(g.c(), next_state, s.recurrent));
        g.run();
    }
    Selection sparse_layer(int layer) {
        ProfileScope scope(&profiler, "layer", "qsa", "qsa", position, layer, ggml_backend_name(backend.get()));
        auto & s = states[layer];
        const std::string p = "blk." + std::to_string(layer) + ".";
        auto g = graph("qsa.prepare", layer);
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
        ggml_tensor * top = nullptr;
        if (complete > 0) {
            auto * cache = ggml_view_2d(g.c(), s.index_cache, c.index_dim, complete, s.index_cache->nb[1], 0);
            // Widen only the compressed index cache, on-device. F32 scoring avoids
            // CPU/CUDA BF16 matvec input-rounding differences at the top-k boundary.
            auto * scores = ggml_mul_mat(g.c(), ggml_cast(g.c(), cache, GGML_TYPE_F32),
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
    Selection full_layer(int layer) {
        ProfileScope scope(&profiler, "layer", "full_attention", "full", position, layer, ggml_backend_name(backend.get()));
        auto & s = states[layer];
        const auto p = "blk." + std::to_string(layer) + ".";
        auto g = graph("full.prepare", layer);
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
    void batch_ffn(ggml_tensor * h, int layer) {
        prefill_memory->protect(layer, {model.group(layer, WeightPart::FeedForward)});
        auto g = graph("prefill.ffn", layer);
        const auto p = "blk." + std::to_string(layer) + ".";
        const int64_t count = h->ne[2];
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
        if (layer + 1 < c.layers) g.before_compute = [this, layer] {
            prefill_memory->prefetch({model.group(layer + 1, WeightPart::PLE), model.group(layer + 1, WeightPart::Attention)});
        };
        g.run();
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
    void batch_gdn(ggml_tensor * h, int layer) {
        prefill_memory->protect(layer, {model.group(layer, WeightPart::Attention)});
        auto g = graph("prefill.gdn", layer);
        auto & state = states[layer];
        const auto p = "blk." + std::to_string(layer) + ".";
        const int count = int(h->ne[2]);
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
    }
    void batch_ple(ggml_tensor * h, int layer, const std::vector<int32_t> & tokens) {
        prefill_memory->protect(layer, {model.group(layer, WeightPart::PLE)});
        auto & state = states[layer];
        auto g = graph("prefill.ple", layer);
        const auto p = "blk." + std::to_string(layer) + ".ple.";
        const int count = int(tokens.size());
        ProfileScope hash(&profiler, "host", "ngram_hash", "prefill.ple", position, layer, "CPU");
        std::vector<int32_t> history_tokens = previous_tokens, rows;
        for (int32_t token : tokens) {
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
    }
    void batch_attention(ggml_tensor * h, int layer, KVExporter & exporter,
                         std::vector<Step> & steps, bool capture) {
        prefill_memory->protect(layer, {model.group(layer, WeightPart::Attention)});
        auto & state = states[layer];
        const int count = int(h->ne[2]), total = position + count;
        const int dim = c.head_dim <= 64 ? 64 : c.head_dim <= 128 ? 128 : 256;
        const int padded = (total + 255) / 256 * 256;
        const int qwidth = c.head_dim * c.q_heads, kvwidth = c.head_dim * c.kv_heads;
        const bool sparse = c.architecture == Architecture::Qwen4Exp;
        const std::string p = "blk." + std::to_string(layer) + ".";
        auto cache = std::make_shared<PrefillTensorStorage>(*prefill_memory, prefill_backend.get());
        auto * keys = cache->tensor(GGML_TYPE_BF16, kvwidth, total);
        auto * values = cache->tensor(GGML_TYPE_BF16, kvwidth, total);
        cache->allocate();
        auto work = std::make_shared<PrefillTensorStorage>(*prefill_memory, prefill_backend.get());
        auto * queries = work->tensor(GGML_TYPE_F32, c.head_dim, c.q_heads, count);
        auto * gates = work->tensor(GGML_TYPE_F32, qwidth, count);
        auto * injection = sparse ? work->tensor(GGML_TYPE_F32, c.branches, count) : nullptr;
        auto * index_queries = sparse ? work->tensor(GGML_TYPE_F32, c.index_dim, c.index_heads, count) : nullptr;
        auto * output = work->tensor(GGML_TYPE_F32, qwidth, count);
        const auto compute_type = options.prefill_precision == PrefillPrecision::FP16 ? GGML_TYPE_F16 : GGML_TYPE_BF16;
        auto * compute_keys = work->tensor(compute_type, dim, padded, c.kv_heads);
        auto * compute_values = work->tensor(compute_type, dim, padded, c.kv_heads);
        work->allocate();
        {
            auto g = graph("prefill.attention.prepare", layer);
            if (position) {
                if (state.attention->size() != position) throw std::logic_error("Prefill KV prefix is inconsistent");
                const size_t bytes = size_t(position) * kvwidth * sizeof(ggml_bf16_t);
                g.put(keys, state.attention->keys(), bytes);
                g.put(values, state.attention->values(), bytes);
            }
            auto mixed = sparse ? batch_mix(g, h, p + "hc_attn")
                : Mixed{norm(g, ggml_reshape_2d(g.c(), h, c.hidden, count), p + "attn_norm"), nullptr};
            auto * qg = mm(g, p + "attn.qg", mixed.value);
            auto * q = ggml_cont(g.c(), ggml_view_3d(g.c(), qg, c.head_dim, c.q_heads, count,
                size_t(2*c.head_dim)*sizeof(float), qg->nb[1], 0));
            auto * gate = ggml_cont(g.c(), ggml_view_3d(g.c(), qg, c.head_dim, c.q_heads, count,
                size_t(2*c.head_dim)*sizeof(float), qg->nb[1], size_t(c.head_dim)*sizeof(float)));
            q = batch_rope(g, norm(g, q, p + "attn.q_norm"), c.head_dim, c.q_heads, count, position);
            auto * k = ggml_reshape_3d(g.c(), mm(g, p + "attn.k", mixed.value), c.head_dim, c.kv_heads, count);
            k = batch_rope(g, norm(g, k, p + "attn.k_norm"), c.head_dim, c.kv_heads, count, position);
            auto * v = mm(g, p + "attn.v", mixed.value);
            g.add(ggml_cpy(g.c(), q, queries));
            g.add(ggml_cpy(g.c(), gate, gates));
            auto suffix = [&](ggml_tensor * t) {
                return ggml_view_2d(g.c(), t, kvwidth, count, t->nb[1], size_t(position)*t->nb[1]);
            };
            g.add(ggml_cpy(g.c(), k, suffix(keys)));
            g.add(ggml_cpy(g.c(), v, suffix(values)));
            if (sparse) {
                g.add(ggml_cpy(g.c(), mixed.inject, injection));
                auto * iq = ggml_reshape_3d(g.c(), mm(g, p + "index.q", mixed.value), c.index_dim, c.index_heads, count);
                iq = batch_rope(g, norm(g, iq, p + "index.q_norm"), c.index_dim, c.index_heads, count, position);
                g.add(ggml_cpy(g.c(), iq, index_queries));
                auto * raw = mm(g, p + "index.k", mixed.value);
                const int old_tail = position % c.block_size;
                if (old_tail) raw = ggml_concat(g.c(), ggml_view_2d(g.c(), state.raw_index,
                    c.index_dim, old_tail, state.raw_index->nb[1], 0), raw, 1);
                const int new_blocks = (old_tail + count) / c.block_size;
                if (new_blocks) {
                    auto * complete = ggml_cont(g.c(), ggml_view_2d(g.c(), raw, c.index_dim,
                        new_blocks*c.block_size, raw->nb[1], 0));
                    complete = ggml_reshape_3d(g.c(), complete, c.index_dim, c.block_size, new_blocks);
                    auto * pooled = ggml_reshape_3d(g.c(), ggml_mean(g.c(), ggml_cont(g.c(),
                        ggml_permute(g.c(), complete, 1, 0, 2, 3))), c.index_dim, 1, new_blocks);
                    pooled = norm(g, pooled, p + "index.k_norm");
                    std::vector<int32_t> starts(new_blocks);
                    for (int j=0; j<new_blocks; ++j) starts[j] = (position/c.block_size+j)*c.block_size;
                    pooled = ggml_rope_ext(g.c(), pooled, g.integers(starts), nullptr, c.rope_dim,
                        GGML_ROPE_TYPE_NEOX, c.context, c.rope_base, 1, 0, 1, 0, 0);
                    auto * destination = ggml_view_2d(g.c(), state.index_cache, c.index_dim, new_blocks,
                        state.index_cache->nb[1], size_t(position/c.block_size)*state.index_cache->nb[1]);
                    g.add(ggml_cpy(g.c(), pooled, destination));
                }
                const int tail = total % c.block_size;
                if (tail) {
                    auto * source = ggml_view_2d(g.c(), raw, c.index_dim, tail, raw->nb[1], size_t(new_blocks*c.block_size)*raw->nb[1]);
                    auto * destination = ggml_view_2d(g.c(), state.raw_index, c.index_dim, tail, state.raw_index->nb[1], 0);
                    g.add(ggml_cpy(g.c(), source, destination));
                }
            }
            g.run();
        }
        // The producer graph is complete. A separate transfer stream and pinned
        // staging preserve the source while GPU attention/FFN continue.
        exporter.enqueue(cache, keys, values, *state.attention, position, count, layer);
        {
            auto g = graph("prefill.attention.layout", layer);
            auto layout = [&](ggml_tensor * source, ggml_tensor * destination) {
                auto * shaped = ggml_reshape_3d(g.c(), source, c.head_dim, c.kv_heads, total);
                auto * wide = ggml_cast(g.c(), shaped, GGML_TYPE_F32);
                auto * permuted = ggml_cont(g.c(), ggml_permute(g.c(), wide, 0, 2, 1, 3));
                auto * padded_kv = ggml_pad(g.c(), permuted, dim-c.head_dim, padded-total, 0, 0);
                g.add(ggml_cpy(g.c(), padded_kv, destination));
            };
            layout(keys, compute_keys); layout(values, compute_values);
            g.run();
        }
        cache.reset(); // Exporter retains it until its asynchronous copy completes.
        const int blocks = total / c.block_size;
        for (int first = 0; first < count; first += options.prefill_attention_tile) {
            const int n = std::min(options.prefill_attention_tile, count-first);
            auto g = graph("prefill.attention", layer, first);
            auto * q = ggml_view_3d(g.c(), queries, c.head_dim, c.q_heads, n,
                queries->nb[1], queries->nb[2], size_t(first)*queries->nb[2]);
            q = ggml_cont(g.c(), ggml_permute(g.c(), q, 0, 2, 1, 3));
            q = ggml_pad(g.c(), q, dim-c.head_dim, 0, 0, 0);
            auto * mask = ggml_fill(g.c(), ggml_new_tensor_2d(g.c(), GGML_TYPE_F32, padded, n), 0);
            mask = ggml_diag_mask_inf(g.c(), mask, position + first);
            ggml_tensor * top = nullptr;
            if (sparse && blocks) {
                auto * iq = ggml_cont(g.c(), ggml_view_3d(g.c(), index_queries, c.index_dim, c.index_heads, n,
                    index_queries->nb[1], index_queries->nb[2], size_t(first)*index_queries->nb[2]));
                auto * compressed = ggml_view_2d(g.c(), state.index_cache, c.index_dim, blocks, state.index_cache->nb[1], 0);
                auto * scores = ggml_mul_mat(g.c(), ggml_cast(g.c(), compressed, GGML_TYPE_F32),
                    ggml_reshape_2d(g.c(), iq, c.index_dim, c.index_heads*n));
                ggml_prec_set_acc(scores, GGML_PREC_F32);
                scores = ggml_reshape_3d(g.c(), ggml_relu(g.c(), scores), blocks, c.index_heads, n);
                scores = ggml_reshape_2d(g.c(), ggml_sum_rows(g.c(), ggml_cont(g.c(), ggml_permute(g.c(), scores, 1, 0, 2, 3))), blocks, n);
                scores = ggml_scale(g.c(), scores, 1.0f/std::sqrt(float(c.index_dim)));
                std::vector<float> completed(n);
                std::vector<int32_t> extra(size_t(n)*c.block_size, -1), mapping(size_t(blocks)*c.block_size);
                std::iota(mapping.begin(), mapping.end(), 0);
                for (int i=0; i<n; ++i) {
                    const int visible = position+first+i+1, full = visible/c.block_size;
                    completed[i] = float(full);
                    for (int j=full*c.block_size; j<visible; ++j) extra[size_t(i)*c.block_size+j-full*c.block_size] = j;
                }
                auto * eligible = ggml_sub(g.c(), ggml_repeat(g.c(), ggml_reshape_2d(g.c(), g.floats(completed), 1, n), scores),
                    ggml_arange(g.c(), 0, float(blocks), 1));
                scores = ggml_add(g.c(), scores, ggml_log(g.c(), ggml_step(g.c(), eligible)));
                top = ggml_cont(g.c(), ggml_top_k(g.c(), scores, std::min(c.top_blocks, blocks)));
                ggml_set_name(top, (p+"index.batch_top_k").c_str());
                mask = ggml_top_k_mask(g.c(), mask, top, scores,
                    ggml_reshape_2d(g.c(), g.integers(mapping), c.block_size, blocks),
                    ggml_reshape_2d(g.c(), g.integers(extra), c.block_size, n));
                if (capture) g.add(top);
            }
            // Padding the query mask does not create additional queries.
            mask = ggml_cast(g.c(), ggml_pad(g.c(), mask, 0, (64-n%64)%64, 0, 0), GGML_TYPE_F16);
            auto * attended = ggml_flash_attn_ext(g.c(), q, compute_keys, compute_values, mask,
                1.0f/std::sqrt(float(c.head_dim)), 0, 0);
            ggml_prec_set_acc(attended, GGML_PREC_F32);
            ggml_set_name(attended, (p+"attention.flash").c_str());
            auto * cropped = ggml_cont(g.c(), ggml_view_3d(g.c(), attended, c.head_dim, c.q_heads, n,
                attended->nb[1], attended->nb[2], 0));
            auto * target = ggml_view_2d(g.c(), output, qwidth, n, output->nb[1], size_t(first)*output->nb[1]);
            g.add(ggml_cpy(g.c(), cropped, target));
            g.run(); ++stats.prefill_gpu_attention_calls;
            if (capture) {
                const auto ids = top ? g.get<int32_t>(top) : std::vector<int32_t>{};
                for (int i=0; i<n; ++i) {
                    const int visible = position+first+i+1;
                    std::vector<int32_t> selected;
                    if (sparse) {
                        std::vector<int32_t> chosen;
                        if (top) for (int j=0; j<top->ne[0]; ++j) {
                            const int id = ids[size_t(i)*top->ne[0]+j];
                            if (id < visible/c.block_size) chosen.push_back(id);
                        }
                        selected = expand_blocks(chosen, visible, c.block_size);
                    } else { selected.resize(visible); std::iota(selected.begin(), selected.end(), 0); }
                    steps[first+i].selections.push_back({layer, std::move(selected)});
                }
            }
        }
        {
            auto g = graph("prefill.attention.finish", layer);
            auto * attended = ggml_mul(g.c(), output, ggml_sigmoid(g.c(), gates));
            auto * projected = mm(g, p+"attn.out", attended);
            auto * result = sparse ? batch_combine(g, h, projected, injection)
                : ggml_add(g.c(), ggml_reshape_2d(g.c(), h, c.hidden, count), projected);
            g.add(ggml_cpy(g.c(), result, h));
            g.before_compute = [this, layer] { prefill_memory->prefetch({model.group(layer, WeightPart::FeedForward)}); };
            g.run();
        }
    }

    std::vector<Step> prefill(const std::vector<int32_t> & tokens, bool capture) {
        if (failed) throw std::runtime_error("Previous execution failed; call reset() before prefill");
        if (tokens.size() > size_t(c.context-position)) throw std::out_of_range("Prompt exceeds context capacity");
        for (auto token : tokens) if (token < 0 || token >= c.vocab) throw std::out_of_range("Prompt token outside vocabulary");
        if (tokens.empty()) return {};
        std::vector<Step> result; result.reserve(tokens.size());
        if (options.backend == "cpu") {
            for (auto token : tokens) result.push_back(decode(token, capture));
            return result;
        }
        if (options.prefill_attention_tile <= 0) throw std::invalid_argument("Prefill attention tile must be positive");
        PrefillMemory memory(model, backend.get(), stats, profiler, options);
        prefill_memory = &memory;
        try {
            prefill_backend = make_backend(options);
            ProfileScope scope(&profiler, "prefill", "layer_major", "prefill", position, -1, ggml_backend_name(prefill_backend.get()));
            {
                KVExporter exporter(options.device, stats, profiler);
                memory.reclaim_completed = [&] { exporter.reap(); };
                memory.wait_transfers = [&] { exporter.finish(); };
                const int count = int(tokens.size());
                for (int i=0; i<count; ++i) result.push_back({position+i, {}, {}});
                const int embedding_group = c.tied_embeddings ? model.group(c.layers, WeightPart::Head) : model.group(-1, WeightPart::Embedding);
                memory.protect(-1, {embedding_group});
                PrefillTensorStorage activations(memory, prefill_backend.get());
                auto * batch = activations.tensor(GGML_TYPE_F32, c.hidden, c.branches, count);
                activations.allocate();
                {
                    auto g = graph("prefill.embedding");
                    auto * embedded = ggml_get_rows(g.c(), weight("token_embd.weight"), g.integers(tokens));
                    embedded = ggml_reshape_3d(g.c(), embedded, c.hidden, 1, count);
                    g.add(ggml_cpy(g.c(), ggml_repeat(g.c(), embedded, batch), batch)); g.run();
                }
                for (int layer=0; layer<c.layers; ++layer) {
                    ProfileScope layer_scope(&profiler, "layer", "batch", "prefill.layer", position, layer, ggml_backend_name(prefill_backend.get()));
                    if (states[layer].ple) batch_ple(batch, layer, tokens);
                    if (c.is_recurrent(layer)) batch_gdn(batch, layer);
                    else batch_attention(batch, layer, exporter, result, capture);
                    batch_ffn(batch, layer);
                    ++stats.prefill_layers;
                }
                memory.protect(c.layers, {model.group(c.layers, WeightPart::Head)});
                {
                    auto g = graph("prefill.head");
                    auto * tail = ggml_view_3d(g.c(), batch, c.hidden, c.branches, 1, batch->nb[1], batch->nb[2], size_t(count-1)*batch->nb[2]);
                    auto project = [&](ggml_tensor * source) {
                        auto mixed = c.architecture == Architecture::Qwen35
                            ? Mixed{norm(g, ggml_reshape_2d(g.c(), source, c.hidden, source->ne[2]), "output_norm.weight"), nullptr}
                            : batch_mix(g, source, "head", false);
                        return mm(g, "output.weight", mixed.value);
                    };
                    auto * logits = capture ? project(batch) : nullptr;
                    if (logits) g.add(logits);
                    // Diagnostics must not change the arithmetic used for sampling.
                    // Quantized GEMM and GEMV can round their inputs differently.
                    auto * last = project(tail);
                    g.add(ggml_cpy(g.c(), last, last_logits));
                    auto * finite = ggml_add(g.c(), ggml_sum(g.c(), ggml_sub(g.c(), batch, batch)),
                        ggml_sum(g.c(), ggml_sub(g.c(), last, last)));
                    if (logits) finite = ggml_add(g.c(), finite, ggml_sum(g.c(), ggml_sub(g.c(), logits, logits)));
                    g.add(finite); g.add(ggml_cpy(g.c(), tail, hidden));
                    g.run();
                    if (!std::isfinite(g.get<float>(finite)[0])) throw std::runtime_error("Non-finite logits or hidden values during prefill");
                    if (capture) {
                        auto all = g.get<float>(logits);
                        for (int i=0; i<count; ++i) result[i].logits.assign(all.begin()+size_t(i)*c.vocab, all.begin()+size_t(i+1)*c.vocab);
                        result.back().logits = g.get<float>(last_logits);
                    }
                }
                exporter.finish();
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
    Step decode(int32_t token, bool capture_logits) {
        if (failed) throw std::runtime_error("Previous decode failed; call reset() before reusing the engine");
        if (token < 0 || token >= c.vocab) throw std::out_of_range("Token ID outside vocabulary");
        if (position >= c.context) throw std::out_of_range("Context capacity exceeded; reset before decoding a new sequence");
        try {
            ProfileScope token_scope(&profiler, "token", "decode", "decode", position, -1, "CPU");
            auto emb = graph("embedding");
            auto * token_embedding = ggml_get_rows(emb.c(), weight("token_embd.weight"), emb.integer(token));
            emb.add(ggml_cpy(emb.c(), ggml_repeat(emb.c(), token_embedding, hidden), hidden));
            emb.run();
            Step step{position, {}, {}};
            for (int layer = 0; layer < c.layers; ++layer) {
                if (states[layer].ple) ple_layer(layer, token);
                if (c.is_recurrent(layer)) linear_layer(layer);
                else step.selections.push_back(c.architecture == Architecture::Qwen35 ? full_layer(layer) : sparse_layer(layer));
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
        model.wait_all();
        for (size_t id=0; id<model.groups.size(); ++id) model.load(int(id));
        ggml_backend_synchronize(backend.get());
        ggml_backend_buffer_clear(state_buffer.get(), 0);
        for (auto & s : states) if (s.attention) s.attention->reset();
        previous_tokens.clear();
        captured_logits.clear();
        sampler->reset();
        position = 0;
        failed = false;
        stats = {};
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
