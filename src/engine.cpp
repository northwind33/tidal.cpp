#include "internal.hpp"
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
    Context ctx = context(4 * 1024 * 1024);
    Buffer storage{nullptr, ggml_backend_buffer_free};
    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 2048, false);
    ggml_backend_t backend;
    Statistics & stats;
    bool device;
    std::vector<std::function<void()>> uploads;
    Graph(ggml_backend_t b, Statistics & s, bool d) : backend(b), stats(s), device(d) {}
    ggml_context * c() { return ctx.get(); }
    void add(ggml_tensor * t) { ggml_build_forward_expand(graph, t); }
    ggml_tensor * integer(int32_t value) {
        auto * t = ggml_new_tensor_1d(c(), GGML_TYPE_I32, 1);
        uploads.emplace_back([this, t, value] { put(t, &value, sizeof(value)); });
        return t;
    }
    ggml_tensor * floats(const std::vector<float> & values) {
        auto * t = ggml_new_tensor_1d(c(), GGML_TYPE_F32, int64_t(values.size()));
        uploads.emplace_back([this, t, values] { put(t, values.data(), values.size() * sizeof(float)); });
        return t;
    }
    void put(ggml_tensor * t, const void * data, size_t size) {
        auto start = Clock::now();
        ggml_backend_tensor_set(t, data, 0, size);
        stats.transfer_ms += elapsed(start);
        if (device) stats.host_to_device_bytes += size;
    }
    template<class T> std::vector<T> get(ggml_tensor * t) {
        std::vector<T> values(size_t(ggml_nelements(t)));
        if (values.size() * sizeof(T) != ggml_nbytes(t) || !ggml_is_contiguous(t))
            throw std::logic_error("Readback must be contiguous with matching type");
        auto start = Clock::now();
        ggml_backend_tensor_get(t, values.data(), 0, ggml_nbytes(t));
        stats.transfer_ms += elapsed(start);
        if (device) stats.device_to_host_bytes += ggml_nbytes(t);
        return values;
    }
    void run() {
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto * node = ggml_graph_node(graph, i);
            if (!ggml_backend_supports_op(backend, node))
                throw std::runtime_error(std::string(ggml_backend_name(backend)) + " does not support " +
                                         ggml_op_name(node->op) + "; no CPU fallback");
        }
        storage.reset(ggml_backend_alloc_ctx_tensors(c(), backend));
        if (!storage) throw std::runtime_error("Graph allocation failed");
        for (auto & upload : uploads) upload();
        auto start = Clock::now();
        const auto status = ggml_backend_graph_compute(backend, graph);
        ggml_backend_synchronize(backend);
        stats.graph_ms += elapsed(start);
        if (status != GGML_STATUS_SUCCESS) throw std::runtime_error("ggml graph execution failed");
        stats.device_graph_nodes += ggml_graph_n_nodes(graph);
    }
};

struct LayerState {
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
    Context state_context;
    Buffer state_buffer{nullptr, ggml_backend_buffer_free};
    ggml_tensor * hidden;
    std::vector<LayerState> states;
    int position = 0;
    bool failed = false;

    Impl(const std::string & path, Options opts)
        : options(std::move(opts)), backend(make_backend(options)), model(path, backend.get()), c(model.config),
          state_context(context(1024 * 1024)), states(c.layers) {
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
        state_buffer.reset(ggml_backend_alloc_ctx_tensors(ctx, backend.get()));
        if (!state_buffer) throw std::runtime_error("State allocation failed");
        reset();
    }
    Graph graph() { return Graph(backend.get(), stats, options.backend == "cuda"); }
    ggml_tensor * weight(const std::string & n) { return model.at(n); }
    ggml_tensor * mm(Graph & g, const std::string & n, ggml_tensor * x) {
        auto * out = ggml_mul_mat(g.c(), weight(n), x);
        // Keep scalar reference comparisons meaningful even when CUDA has a lower-precision matmul path.
        ggml_prec_set_acc(out, GGML_PREC_F32);
        return out;
    }
    ggml_tensor * norm(Graph & g, ggml_tensor * x, const std::string & n) {
        return ggml_mul(g.c(), ggml_rms_norm(g.c(), x, c.eps), weight(n));
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
        auto * v = ggml_mul(g.c(), ggml_silu(g.c(), mm(g, p + "ffn.gate", mixed.value)),
                           mm(g, p + "ffn.up", mixed.value));
        return combine(g, h, mm(g, p + "ffn.down", v), mixed.inject);
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
    void linear_layer(int layer) {
        auto g = graph();
        auto & s = states[layer];
        const std::string p = "blk." + std::to_string(layer) + ".";
        auto mixed = mix(g, hidden, p + "hc_attn");
        auto * qkv = mm(g, p + "gdn.qkv", mixed.value);
        const int kd = c.gdn_key_heads * c.gdn_dim, vd = c.gdn_value_heads * c.gdn_dim;
        const int channels = 2 * kd + vd;
        auto * row = ggml_reshape_2d(g.c(), qkv, 1, channels);
        auto * history = ggml_concat(g.c(), s.conv, row, 0);
        auto * convolved = ggml_silu(g.c(), ggml_ssm_conv(g.c(), history, weight(p + "gdn.conv")));
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
        auto & s = states[layer];
        const std::string p = "blk." + std::to_string(layer) + ".";
        auto g = graph();
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
            scores = ggml_relu(g.c(), scores);
            scores = ggml_sum_rows(g.c(), ggml_cont(g.c(), ggml_transpose(g.c(), scores)));
            scores = ggml_scale(g.c(), ggml_reshape_1d(g.c(), scores, complete), 1.0f / std::sqrt(float(c.index_dim)));
            top = ggml_cont(g.c(), ggml_top_k(g.c(), scores, std::min(c.top_blocks, complete)));
            g.add(top);
        }
        g.run();
        auto blocks = top ? g.get<int32_t>(top) : std::vector<int32_t>{};
        auto selected = expand_blocks(blocks, visible, c.block_size);
        auto queries = g.get<float>(q), keys = g.get<float>(k), values = g.get<float>(v);
        auto start = Clock::now();
        s.attention->append(keys, values);
        auto output = s.attention->attend(queries, selected);
        stats.attention_ms += elapsed(start);
        ++stats.sparse_attention_calls;

        auto post = graph();
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
            auto emb = graph();
            auto * token_embedding = ggml_get_rows(emb.c(), weight("token_embd.weight"), emb.integer(token));
            emb.add(ggml_cpy(emb.c(), ggml_repeat(emb.c(), token_embedding, hidden), hidden));
            emb.run();
            Step step{position, {}, {}};
            for (int layer = 0; layer < c.layers; ++layer) {
                if ((layer + 1) % 4) linear_layer(layer);
                else step.selections.push_back(sparse_layer(layer));
            }
            auto head = graph();
            auto mixed = mix(head, hidden, "head", false);
            auto * logits = mm(head, "output.weight", mixed.value);
            head.add(logits);
            head.run();
            step.logits = head.get<float>(logits);
            if (!std::all_of(step.logits.begin(), step.logits.end(), [](float f) { return std::isfinite(f); }))
                throw std::runtime_error("Non-finite logits");
            ++position;
            return step;
        } catch (...) { failed = true; throw; }
    }
    void reset() {
        ggml_backend_synchronize(backend.get());
        ggml_backend_buffer_clear(state_buffer.get(), 0);
        for (auto & s : states) if (s.attention) s.attention->reset();
        position = 0;
        failed = false;
        stats = {};
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
std::string Engine::backend_name() const { return ggml_backend_name(impl_->backend.get()); }
} // namespace tidal
