#include "internal.hpp"
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
    Statistics & stats;
    bool device;
    std::vector<std::function<void()>> uploads;
    Graph(ggml_backend_t b, Statistics & s, bool d, Profiler & p,
          std::string ph, int pos, int l)
        : profiler(p), phase(std::move(ph)), backend_label(ggml_backend_name(b)), position(pos), layer(l),
          build_scope(&p, "host", "graph_build", phase, pos, l, "CPU"),
          backend(b), stats(s), device(d) {}
    ~Graph() {
        ProfileScope release(&profiler, "host", "graph_free", phase, position, layer, "CPU");
        storage.reset();
        ctx.reset();
    }
    ggml_context * c() { return ctx.get(); }
    void add(ggml_tensor * t) { ggml_build_forward_expand(graph, t); }
    ggml_tensor * integer(int32_t value) {
        return integers({value});
    }
    ggml_tensor * integers(const std::vector<int32_t> & values) {
        auto * t = ggml_new_tensor_1d(c(), GGML_TYPE_I32, int64_t(values.size()));
        uploads.emplace_back([this, t, values] { put(t, values.data(), values.size() * sizeof(int32_t)); });
        return t;
    }
    ggml_tensor * floats(const std::vector<float> & values) {
        auto * t = ggml_new_tensor_1d(c(), GGML_TYPE_F32, int64_t(values.size()));
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
        storage.reset(ggml_backend_alloc_ctx_tensors(c(), backend));
        if (!storage) throw std::runtime_error("Graph allocation failed");
        allocation.finish();
        for (auto & upload : uploads) upload();
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
    std::vector<LayerState> states;
    std::vector<int32_t> previous_tokens;
    int position = 0;
    bool failed = false;

    Impl(const std::string & path, Options opts)
        : options(std::move(opts)), backend(make_backend(options)), model(path, backend.get()), c(model.config),
          profiler(options.profile, options.nvtx, options.profile_event_limit), state_context(context(1024 * 1024)), states(c.layers) {
        auto * ctx = state_context.get();
        hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.hidden, c.branches);
        for (int i = 0; i < c.layers; ++i) {
            auto & s = states[i];
            if ((i + 1) % 4) {
                const int channels = (2 * c.gdn_key_heads + c.gdn_value_heads) * c.gdn_dim;
                s.conv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.conv_kernel - 1, channels);
                s.recurrent = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                    c.gdn_dim, c.gdn_dim, c.gdn_value_heads, 1);
            } else {
                s.raw_index = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.index_dim, c.block_size);
                s.index_cache = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, c.index_dim,
                                                  std::max(1, c.context / c.block_size));
                s.gate = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.q_heads * c.head_dim);
                s.inject = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.branches);
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
        reset();
    }
    Graph graph(const std::string & phase, int layer = -1) {
        return Graph(backend.get(), stats, options.backend == "cuda", profiler, phase, position, layer);
    }
    ggml_tensor * weight(const std::string & n) { return model.at(n); }
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
        auto mixed = mix(g, hidden, p + "hc_attn");
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
        q = repeat_gdn_heads(g, ggml_scale(g.c(), ggml_rms_norm(g.c(), q, c.eps / c.gdn_dim), scale));
        k = repeat_gdn_heads(g, ggml_scale(g.c(), ggml_rms_norm(g.c(), k, c.eps / c.gdn_dim), scale));
        auto * a = ggml_softplus(g.c(), ggml_add(g.c(), mm(g, p + "gdn.a", mixed.value), weight(p + "gdn.dt_bias")));
        auto * decay = ggml_neg(g.c(), ggml_mul(g.c(), ggml_exp(g.c(), weight(p + "gdn.a_log")), a));
        auto * beta = ggml_sigmoid(g.c(), mm(g, p + "gdn.b", mixed.value));
        decay = ggml_reshape_4d(g.c(), decay, 1, c.gdn_value_heads, 1, 1);
        beta = ggml_reshape_4d(g.c(), beta, 1, c.gdn_value_heads, 1, 1);
        auto * packed = ggml_gated_delta_net(g.c(), q, k, v, decay, beta, s.recurrent, 1);
        ggml_set_name(packed, (p + "gdn.recurrence").c_str());
        auto * output = ggml_reshape_2d(g.c(), ggml_view_1d(g.c(), packed, vd, 0), c.gdn_dim, c.gdn_value_heads);
        auto * z = ggml_reshape_2d(g.c(), mm(g, p + "gdn.z", mixed.value), c.gdn_dim, c.gdn_value_heads);
        output = ggml_mul(g.c(), norm(g, output, p + "gdn.norm"), ggml_silu(g.c(), z));
        output = mm(g, p + "gdn.out", ggml_reshape_1d(g.c(), output, vd));
        auto * h = combine(g, hidden, output, mixed.inject);
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
    Step decode(int32_t token) {
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
                if ((layer + 1) % 4) linear_layer(layer);
                else step.selections.push_back(sparse_layer(layer));
            }
            auto head = graph("head");
            auto mixed = mix(head, hidden, "head", false);
            auto * logits = mm(head, "output.weight", mixed.value);
            head.add(logits);
            head.run();
            step.logits = head.get<float>(logits);
            if (!std::all_of(step.logits.begin(), step.logits.end(), [](float f) { return std::isfinite(f); }))
                throw std::runtime_error("Non-finite logits");
            if (!c.ple.layers.empty()) {
                previous_tokens.push_back(token);
                if (previous_tokens.size() >= size_t(c.ple.ngram_size)) previous_tokens.erase(previous_tokens.begin());
            }
            ++position;
            return step;
        } catch (...) { failed = true; throw; }
    }
    void reset() {
        ggml_backend_synchronize(backend.get());
        ggml_backend_buffer_clear(state_buffer.get(), 0);
        for (auto & s : states) if (s.attention) s.attention->reset();
        previous_tokens.clear();
        position = 0;
        failed = false;
        stats = {};
        profiler.clear();
    }
};

Engine::Engine(const std::string & path, Options options) : impl_(std::make_unique<Impl>(path, std::move(options))) {}
Engine::~Engine() = default;
Step Engine::decode(int32_t token) { return impl_->decode(token); }
std::vector<Step> Engine::prefill(const std::vector<int32_t> & tokens) {
    if (tokens.size() > size_t(impl_->c.context - impl_->position)) throw std::out_of_range("Prompt exceeds context capacity");
    for (auto t : tokens) if (t < 0 || t >= impl_->c.vocab) throw std::out_of_range("Prompt token outside vocabulary");
    std::vector<Step> result;
    result.reserve(tokens.size());
    for (int32_t t : tokens) result.push_back(decode(t));
    return result;
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
