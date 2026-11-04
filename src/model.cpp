#include "internal.hpp"

#include <cmath>
#include <fstream>
#include <limits>

namespace tidal {
namespace {
struct IntField { const char * name; int Config::* value; };
constexpr IntField fields[] = {
    {"hidden", &Config::hidden}, {"ffn", &Config::ffn}, {"vocab", &Config::vocab},
    {"layers", &Config::layers}, {"context", &Config::context},
    {"branches", &Config::branches}, {"rank", &Config::rank},
    {"q_heads", &Config::q_heads}, {"kv_heads", &Config::kv_heads},
    {"head_dim", &Config::head_dim}, {"rope_dim", &Config::rope_dim},
    {"gdn_key_heads", &Config::gdn_key_heads}, {"gdn_value_heads", &Config::gdn_value_heads},
    {"gdn_dim", &Config::gdn_dim}, {"conv_kernel", &Config::conv_kernel},
    {"index_heads", &Config::index_heads}, {"index_dim", &Config::index_dim},
    {"block_size", &Config::block_size}, {"top_blocks", &Config::top_blocks},
};
int64_t key(const gguf_context * f, const std::string & name, gguf_type type) {
    const auto id = gguf_find_key(f, name.c_str());
    if (id < 0 || gguf_get_kv_type(f, id) != type)
        throw std::runtime_error("Missing or wrongly typed GGUF metadata: " + name);
    return id;
}
}

void Config::validate() const {
    for (auto f : fields) {
        if (this->*(f.value) <= 0)
            throw std::invalid_argument(std::string(f.name) + " must be positive");
    }
    // Bound allocations and products before constructing ggml tensors. This is a tiny-model schema.
    if (hidden > 1024 || ffn > 4096 || vocab > 65536 || layers > 32 || context > 1048576 ||
        branches > 8 || rank > 256 || q_heads > 32 || kv_heads > 32 || head_dim > 256 ||
        gdn_key_heads > 32 || gdn_value_heads > 32 || index_heads > 16 || index_dim > 256 ||
        conv_kernel > 16 || block_size > 32 || top_blocks > 4096)
        throw std::invalid_argument("Configuration exceeds the tiny-engine limits");
    if (layers % 4 || branches < 2 || conv_kernel < 2 || block_size < 2 || q_heads % kv_heads || gdn_value_heads % gdn_key_heads ||
        rope_dim % 2 || rope_dim > head_dim || rope_dim > index_dim ||
        (gdn_dim != 32 && gdn_dim != 64 && gdn_dim != 128) ||
        hidden % 32 || ffn % 32 || rank % 32 || (q_heads * head_dim) % 32)
        throw std::invalid_argument("Unsupported layer/head/rotary/quantization dimensions");
    if (!std::isfinite(eps) || eps <= 0 || !std::isfinite(rope_base) || rope_base <= 1)
        throw std::invalid_argument("Invalid RMS epsilon or RoPE base");
    validate_ple(*this);
}

std::vector<TensorSpec> tensor_specs(const Config & c) {
    c.validate();
    std::vector<TensorSpec> specs;
    auto add = [&](std::string n, int64_t in, int64_t out, TensorSpec::Kind k = TensorSpec::Matrix) {
        specs.push_back({std::move(n), in, out, k});
    };
    auto hc = [&](const std::string & p, bool inject) {
        add(p + ".norm", c.hidden, c.branches, TensorSpec::Norm);
        add(p + ".down", c.hidden * c.branches, c.rank);
        add(p + ".up", c.rank, c.hidden * c.branches);
        if (inject) add(p + ".inject", c.hidden * c.branches, c.branches);
    };
    add("token_embd.weight", c.hidden, c.vocab, TensorSpec::Embedding);
    hc("head", false);
    add("output.weight", c.hidden, c.vocab);
    for (int layer = 0; layer < c.layers; ++layer) {
        const std::string p = "blk." + std::to_string(layer) + ".";
        hc(p + "hc_attn", true);
        hc(p + "hc_ffn", true);
        add(p + "ffn.gate", c.hidden, c.ffn);
        add(p + "ffn.up", c.hidden, c.ffn);
        add(p + "ffn.down", c.ffn, c.hidden);
        if ((layer + 1) % 4) {
            int k = c.gdn_key_heads * c.gdn_dim, v = c.gdn_value_heads * c.gdn_dim;
            add(p + "gdn.qkv", c.hidden, 2 * k + v);
            add(p + "gdn.z", c.hidden, v);
            add(p + "gdn.a", c.hidden, c.gdn_value_heads);
            add(p + "gdn.b", c.hidden, c.gdn_value_heads);
            add(p + "gdn.conv", c.conv_kernel, 2 * k + v, TensorSpec::Conv);
            add(p + "gdn.a_log", c.gdn_value_heads, 1, TensorSpec::ALog);
            add(p + "gdn.dt_bias", c.gdn_value_heads, 1, TensorSpec::Bias);
            add(p + "gdn.norm", c.gdn_dim, 1, TensorSpec::Norm);
            add(p + "gdn.out", v, c.hidden);
        } else {
            add(p + "attn.qg", c.hidden, c.q_heads * c.head_dim * 2);
            add(p + "attn.k", c.hidden, c.kv_heads * c.head_dim);
            add(p + "attn.v", c.hidden, c.kv_heads * c.head_dim);
            add(p + "attn.out", c.q_heads * c.head_dim, c.hidden);
            add(p + "attn.q_norm", c.head_dim, 1, TensorSpec::Norm);
            add(p + "attn.k_norm", c.head_dim, 1, TensorSpec::Norm);
            add(p + "index.q", c.hidden, c.index_heads * c.index_dim);
            add(p + "index.k", c.hidden, c.index_dim);
            add(p + "index.q_norm", c.index_dim, 1, TensorSpec::Norm);
            add(p + "index.k_norm", c.index_dim, 1, TensorSpec::Norm);
        }
    }
    // Append optional tensors so PLE does not change seeded base-model weights.
    for (const auto & l : c.ple.layers) {
        const std::string p = "blk." + std::to_string(l.layer) + ".ple.";
        add(p + "embedding", c.ple.embedding_dim / c.ple.heads(), l.vocab, TensorSpec::Lookup);
        add(p + "key", c.ple.embedding_dim, c.hidden * c.branches);
        add(p + "value", c.ple.embedding_dim, c.hidden);
        for (const auto * name : {"norm_key", "norm_query", "norm_conv"})
            add(p + name, c.hidden, c.branches, TensorSpec::Norm);
        add(p + "conv", c.ple.conv_kernel, c.hidden * c.branches, TensorSpec::Conv);
    }
    return specs;
}

void write_config(gguf_context * f, const Config & c) {
    c.validate();
    gguf_set_val_str(f, "general.architecture", architecture);
    gguf_set_val_u32(f, "tidal.version", format_version);
    for (auto field : fields)
        gguf_set_val_u32(f, (std::string("tidal.") + field.name).c_str(), c.*(field.value));
    gguf_set_val_f32(f, "tidal.eps", c.eps);
    gguf_set_val_f32(f, "tidal.rope_base", c.rope_base);
    write_ple(f, c.ple);
}

Config read_config(const gguf_context * f) {
    if (std::string(gguf_get_val_str(f, key(f, "general.architecture", GGUF_TYPE_STRING))) != architecture)
        throw std::runtime_error("Expected tidal-qwen4exp-dense; production Qwen checkpoints are not supported");
    if (gguf_get_val_u32(f, key(f, "tidal.version", GGUF_TYPE_UINT32)) != format_version)
        throw std::runtime_error("Unsupported tidal GGUF schema version");
    Config c;
    for (auto field : fields) {
        auto v = gguf_get_val_u32(f, key(f, std::string("tidal.") + field.name, GGUF_TYPE_UINT32));
        if (v > uint32_t(std::numeric_limits<int>::max())) throw std::runtime_error("Metadata overflow");
        c.*(field.value) = int(v);
    }
    c.eps = gguf_get_val_f32(f, key(f, "tidal.eps", GGUF_TYPE_FLOAT32));
    c.rope_base = gguf_get_val_f32(f, key(f, "tidal.rope_base", GGUF_TYPE_FLOAT32));
    read_ple(f, c);
    c.validate();
    return c;
}

Model::Model(const std::string & path, ggml_backend_t backend) {
    ggml_context * raw = nullptr;
    Gguf file(gguf_init_from_file(path.c_str(), {true, &raw}), gguf_free);
    ctx.reset(raw);
    if (!file || !ctx) throw std::runtime_error("Cannot read GGUF: " + path);
    config = read_config(file.get());
    const auto specs = tensor_specs(config);
    if (gguf_get_n_tensors(file.get()) != int64_t(specs.size()))
        throw std::runtime_error("Unexpected number of model tensors");
    for (const auto & spec : specs) {
        auto * t = ggml_get_tensor(ctx.get(), spec.name.c_str());
        if (!t || t->ne[0] != spec.columns || t->ne[1] != spec.rows || t->ne[2] != 1 || t->ne[3] != 1)
            throw std::runtime_error("Missing or invalid tensor shape: " + spec.name);
        const bool matrix = spec.kind == TensorSpec::Matrix || spec.kind == TensorSpec::Lookup;
        const bool embedding = spec.kind == TensorSpec::Embedding;
        const bool supported = t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16 ||
            t->type == GGML_TYPE_BF16 || t->type == GGML_TYPE_Q4_0 || t->type == GGML_TYPE_Q8_0;
        if (!supported || (!matrix && !embedding && t->type != GGML_TYPE_F32) ||
            (embedding && t->type != GGML_TYPE_BF16))
            throw std::runtime_error("Unsupported tensor type: " + spec.name);
        weights.emplace(spec.name, t);
    }
    buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (!buffer) throw std::runtime_error("Cannot allocate model weights");
    ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::ifstream input(path, std::ios::binary);
    input.seekg(0, std::ios::end);
    const auto length = input.tellg();
    if (length < 0) throw std::runtime_error("Cannot determine model file length");
    for (const auto & [name, tensor] : weights) {
        const auto id = gguf_find_tensor(file.get(), name.c_str());
        const uint64_t offset = gguf_get_data_offset(file.get()) + gguf_get_tensor_offset(file.get(), id);
        const auto size = ggml_nbytes(tensor);
        if (offset > uint64_t(length) || size > uint64_t(length) - offset)
            throw std::runtime_error("Truncated GGUF tensor: " + name);
        std::vector<char> data(size);
        input.seekg(std::streamoff(offset));
        input.read(data.data(), std::streamsize(size));
        if (!input) throw std::runtime_error("Failed to read tensor: " + name);
        ggml_backend_tensor_set(tensor, data.data(), 0, size);
    }
}
} // namespace tidal
