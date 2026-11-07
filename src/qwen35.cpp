#include "internal.hpp"

#include <algorithm>
#include <cmath>
#include <climits>
#include <set>

namespace tidal {
namespace {
int64_t required(const gguf_context * f, const std::string & name, gguf_type type) {
    const auto id = gguf_find_key(f, name.c_str());
    if (id < 0 || gguf_get_kv_type(f, id) != type)
        throw std::runtime_error("Missing or wrongly typed Qwen3.5 metadata: " + name);
    return id;
}
int integer(const gguf_context * f, const char * suffix, int fallback = -1) {
    const std::string name = std::string("qwen35.") + suffix;
    if (fallback >= 0 && gguf_find_key(f, name.c_str()) < 0) return fallback;
    const auto value = gguf_get_val_u32(f, required(f, name, GGUF_TYPE_UINT32));
    if (value > INT_MAX) throw std::runtime_error("Qwen3.5 metadata overflow: " + name);
    return int(value);
}
}

void validate_qwen35(const Config & c) {
    if (c.hidden < 32 || c.hidden > 16384 || c.hidden % 32 || c.ffn < 32 || c.ffn > 131072 || c.ffn % 32 ||
        c.vocab < 1 || c.vocab > 1048576 || c.layers < 1 || c.layers > 128 ||
        c.context < 1 || c.context > 1048576 || c.branches != 1 ||
        c.q_heads < 1 || c.q_heads > 128 || c.kv_heads < 1 || c.kv_heads > 128 || c.q_heads % c.kv_heads ||
        c.head_dim < 2 || c.head_dim > 256 || c.rope_dim < 2 || c.rope_dim > c.head_dim || c.rope_dim % 2 ||
        c.gdn_key_heads < 1 || c.gdn_key_heads > 128 || c.gdn_value_heads < 1 || c.gdn_value_heads > 128 ||
        c.gdn_value_heads % c.gdn_key_heads || (c.gdn_dim != 32 && c.gdn_dim != 64 && c.gdn_dim != 128) ||
        c.conv_kernel < 2 || c.conv_kernel > 16 || c.mtp_layers < 0 || c.mtp_layers > 8 ||
        c.recurrent_layers.size() != size_t(c.layers) || c.experts || !c.ple.layers.empty() ||
        c.eos_token < -1 || c.eos_token >= c.vocab)
        throw std::invalid_argument("Unsupported Qwen3.5 dimensions or layer configuration");
    if (!std::isfinite(c.eps) || c.eps <= 0 || !std::isfinite(c.rope_base) || c.rope_base <= 1)
        throw std::invalid_argument("Invalid Qwen3.5 normalization or RoPE parameters");
}

Config read_qwen35_config(const gguf_context * f) {
    Config c;
    c.architecture = Architecture::Qwen35;
    c.branches = 1;
    c.hidden = integer(f, "embedding_length");
    c.ffn = integer(f, "feed_forward_length");
    c.context = integer(f, "context_length");
    const int total_layers = integer(f, "block_count");
    c.mtp_layers = integer(f, "nextn_predict_layers", 0);
    if (total_layers < 1 || total_layers > 136 || c.mtp_layers >= total_layers)
        throw std::runtime_error("Invalid Qwen3.5 block/MTP counts");
    c.layers = total_layers - c.mtp_layers;
    c.q_heads = integer(f, "attention.head_count");
    c.kv_heads = integer(f, "attention.head_count_kv");
    c.head_dim = integer(f, "attention.key_length");
    if (integer(f, "attention.value_length") != c.head_dim)
        throw std::runtime_error("Qwen3.5 requires equal key/value head widths");
    c.rope_dim = integer(f, "rope.dimension_count");
    c.conv_kernel = integer(f, "ssm.conv_kernel");
    c.gdn_dim = integer(f, "ssm.state_size");
    c.gdn_key_heads = integer(f, "ssm.group_count");
    c.gdn_value_heads = integer(f, "ssm.time_step_rank");
    if (int64_t(integer(f, "ssm.inner_size")) != int64_t(c.gdn_dim) * c.gdn_value_heads)
        throw std::runtime_error("Qwen3.5 GDN key/value state widths must match");
    c.eps = gguf_get_val_f32(f, required(f, "qwen35.attention.layer_norm_rms_epsilon", GGUF_TYPE_FLOAT32));
    c.rope_base = gguf_get_val_f32(f, required(f, "qwen35.rope.freq_base", GGUF_TYPE_FLOAT32));
    const auto sections = required(f, "qwen35.rope.dimension_sections", GGUF_TYPE_ARRAY);
    if (gguf_get_arr_type(f, sections) != GGUF_TYPE_INT32 || gguf_get_arr_n(f, sections) != 4)
        throw std::runtime_error("Invalid Qwen3.5 MRoPE sections");
    const auto * values = static_cast<const int32_t *>(gguf_get_arr_data(f, sections));
    int64_t sum = 0;
    for (int i = 0; i < 4; ++i) {
        if (values[i] < 0) throw std::runtime_error("Negative Qwen3.5 MRoPE section");
        sum += values[i];
    }
    if (sum != c.rope_dim / 2 || values[3] != 0)
        throw std::runtime_error("Unsupported Qwen3.5 MRoPE layout");
    // Text positions are equal on all three MRoPE axes, reducing to partial NeoX RoPE.
    const auto scaling = gguf_find_key(f, "qwen35.rope.scaling.type");
    if (scaling >= 0 && (gguf_get_kv_type(f, scaling) != GGUF_TYPE_STRING ||
                        std::string(gguf_get_val_str(f, scaling)) != "none"))
        throw std::runtime_error("Qwen3.5 scaled RoPE is not supported");
    const auto factor = gguf_find_key(f, "qwen35.rope.scaling.factor");
    if (factor >= 0 && (gguf_get_kv_type(f, factor) != GGUF_TYPE_FLOAT32 ||
                       gguf_get_val_f32(f, factor) != 1.0f))
        throw std::runtime_error("Qwen3.5 scaled RoPE is not supported");
    const auto recurrent = gguf_find_key(f, "qwen35.attention.recurrent_layers");
    c.recurrent_layers.resize(c.layers);
    if (recurrent >= 0) {
        if (gguf_get_kv_type(f, recurrent) != GGUF_TYPE_ARRAY || gguf_get_arr_type(f, recurrent) != GGUF_TYPE_BOOL ||
            gguf_get_arr_n(f, recurrent) != size_t(total_layers))
            throw std::runtime_error("Invalid Qwen3.5 recurrent layer mask");
        const auto * mask = static_cast<const bool *>(gguf_get_arr_data(f, recurrent));
        for (int i = 0; i < total_layers; ++i) {
            if (i < c.layers) c.recurrent_layers[i] = mask[i];
            else if (mask[i]) throw std::runtime_error("Qwen3.5 MTP layers must use full attention");
        }
    } else {
        const int interval = integer(f, "full_attention_interval", 4);
        if (interval < 1 || interval > 128) throw std::runtime_error("Invalid full-attention interval");
        for (int i = 0; i < c.layers; ++i) c.recurrent_layers[i] = (i + 1) % interval != 0;
    }
    const auto embedding = gguf_find_tensor(f, "token_embd.weight");
    if (embedding < 0) throw std::runtime_error("Missing Qwen3.5 token embeddings");
    const auto * shape = gguf_get_tensor_ne(f, embedding);
    if (shape[1] < 1 || shape[1] > 1048576) throw std::runtime_error("Invalid Qwen3.5 vocabulary size");
    c.vocab = int(shape[1]);
    const auto eos = gguf_find_key(f, "tokenizer.ggml.eos_token_id");
    if (eos >= 0) {
        const auto id = gguf_get_val_u32(f, required(f, "tokenizer.ggml.eos_token_id", GGUF_TYPE_UINT32));
        if (id >= uint32_t(c.vocab)) throw std::runtime_error("Invalid Qwen3.5 EOS token");
        c.eos_token = int(id);
    }
    c.tied_embeddings = gguf_find_tensor(f, "output.weight") < 0;
    c.validate();
    return c;
}

std::vector<TensorSpec> qwen35_tensor_specs(const Config & c) {
    std::vector<TensorSpec> result;
    auto add = [&](std::string name, std::string source, int64_t in, int64_t out = 1,
                   TensorSpec::Kind kind = TensorSpec::Matrix) {
        result.push_back({std::move(name), in, out, kind, 1, std::move(source)});
    };
    add("token_embd.weight", "token_embd.weight", c.hidden, c.vocab, TensorSpec::Embedding);
    add("output_norm.weight", "output_norm.weight", c.hidden, 1, TensorSpec::Norm);
    if (!c.tied_embeddings) add("output.weight", "output.weight", c.hidden, c.vocab);
    for (int i = 0; i < c.layers; ++i) {
        const auto p = "blk." + std::to_string(i) + ".";
        add(p + "attn_norm", p + "attn_norm.weight", c.hidden, 1, TensorSpec::Norm);
        add(p + "ffn_norm", p + "post_attention_norm.weight", c.hidden, 1, TensorSpec::Norm);
        for (const auto * name : {"gate", "up", "down"}) {
            const bool down = std::string(name) == "down";
            add(p + "ffn." + name, p + "ffn_" + name + ".weight", down ? c.ffn : c.hidden, down ? c.hidden : c.ffn);
        }
        if (c.is_recurrent(i)) {
            const int k = c.gdn_key_heads * c.gdn_dim, v = c.gdn_value_heads * c.gdn_dim;
            add(p + "gdn.qkv", p + "attn_qkv.weight", c.hidden, 2*k + v);
            add(p + "gdn.z", p + "attn_gate.weight", c.hidden, v);
            add(p + "gdn.a", p + "ssm_alpha.weight", c.hidden, c.gdn_value_heads);
            add(p + "gdn.b", p + "ssm_beta.weight", c.hidden, c.gdn_value_heads);
            add(p + "gdn.conv", p + "ssm_conv1d.weight", c.conv_kernel, 2*k + v, TensorSpec::Conv);
            add(p + "gdn.a_log", p + "ssm_a", c.gdn_value_heads, 1, TensorSpec::ALog);
            add(p + "gdn.dt_bias", p + "ssm_dt.bias", c.gdn_value_heads, 1, TensorSpec::Bias);
            add(p + "gdn.norm", p + "ssm_norm.weight", c.gdn_dim, 1, TensorSpec::Norm);
            add(p + "gdn.out", p + "ssm_out.weight", v, c.hidden);
        } else {
            add(p + "attn.qg", p + "attn_q.weight", c.hidden, 2*c.q_heads*c.head_dim);
            add(p + "attn.k", p + "attn_k.weight", c.hidden, c.kv_heads*c.head_dim);
            add(p + "attn.v", p + "attn_v.weight", c.hidden, c.kv_heads*c.head_dim);
            add(p + "attn.out", p + "attn_output.weight", c.q_heads*c.head_dim, c.hidden);
            add(p + "attn.q_norm", p + "attn_q_norm.weight", c.head_dim, 1, TensorSpec::Norm);
            add(p + "attn.k_norm", p + "attn_k_norm.weight", c.head_dim, 1, TensorSpec::Norm);
        }
    }
    return result;
}

bool qwen35_mtp_tensor(const Config & c, const std::string & name) {
    if (c.architecture != Architecture::Qwen35) return false;
    const std::set<std::string> suffixes = {"attn_norm.weight", "post_attention_norm.weight", "attn_q.weight",
        "attn_k.weight", "attn_v.weight", "attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight",
        "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight", "nextn.eh_proj.weight", "nextn.enorm.weight",
        "nextn.hnorm.weight", "nextn.embed_tokens.weight", "nextn.shared_head_head.weight", "nextn.shared_head_norm.weight"};
    for (int i = c.layers; i < c.layers + c.mtp_layers; ++i) {
        const auto prefix = "blk." + std::to_string(i) + ".";
        if (name.starts_with(prefix)) return suffixes.contains(name.substr(prefix.size()));
    }
    return false;
}
} // namespace tidal
