#include "internal.hpp"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
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
    // Bound dimension products before constructing tensors. Weight sizes use
    // int64_t; actual allocation is determined by the file and backend memory.
    if (hidden > 16384 || ffn > 131072 || vocab > 262144 || layers > 128 || context > 1048576 ||
        branches > 8 || rank > 2048 || q_heads > 128 || kv_heads > 128 || head_dim > 256 ||
        gdn_key_heads > 128 || gdn_value_heads > 128 || index_heads > 64 || index_dim > 256 ||
        conv_kernel > 16 || block_size > 128 || top_blocks > 4096)
        throw std::invalid_argument("Configuration exceeds supported dimension limits");
    if (layers % 4 || branches < 2 || conv_kernel < 2 || block_size < 2 || q_heads % kv_heads || gdn_value_heads % gdn_key_heads ||
        rope_dim % 2 || rope_dim > head_dim || rope_dim > index_dim ||
        (gdn_dim != 32 && gdn_dim != 64 && gdn_dim != 128) ||
        hidden % 32 || ffn % 32 || rank % 32 || (q_heads * head_dim) % 32)
        throw std::invalid_argument("Unsupported layer/head/rotary/quantization dimensions");
    if (!std::isfinite(eps) || eps <= 0 || !std::isfinite(rope_base) || rope_base <= 1)
        throw std::invalid_argument("Invalid RMS epsilon or RoPE base");
    if (experts < 0 || experts > 256 || experts_used < 0 || expert_ffn < 0 || shared_ffn < 0 ||
        (experts == 0 && (experts_used || expert_ffn || shared_ffn)) ||
        (experts && (experts_used < 1 || experts_used > experts || expert_ffn < 32 ||
                     expert_ffn > 65536 || shared_ffn > 65536 || expert_ffn % 32 || shared_ffn % 32)))
        throw std::invalid_argument("Invalid MoE expert count, top-k or FFN dimensions");
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
        if (!c.experts) {
            add(p + "ffn.gate", c.hidden, c.ffn);
            add(p + "ffn.up", c.hidden, c.ffn);
            add(p + "ffn.down", c.ffn, c.hidden);
        } else {
            add(p + "ffn.router", c.hidden, c.experts, TensorSpec::Router);
            for (auto name : {"gate_exps", "up_exps", "down_exps"}) {
                const bool down = std::string(name) == "down_exps";
                add(p + "ffn." + name, down ? c.expert_ffn : c.hidden, down ? c.hidden : c.expert_ffn, TensorSpec::Expert);
                specs.back().slices = c.experts;
            }
            if (c.shared_ffn) {
                add(p + "ffn.shared.gate", c.hidden, c.shared_ffn);
                add(p + "ffn.shared.up", c.hidden, c.shared_ffn);
                add(p + "ffn.shared.down", c.shared_ffn, c.hidden);
                add(p + "ffn.shared_gate", c.hidden, 1, TensorSpec::Router);
            }
        }
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
    gguf_set_val_str(f, "general.architecture", c.experts ? moe_architecture : architecture);
    gguf_set_val_u32(f, "tidal.version", format_version);
    for (auto field : fields)
        gguf_set_val_u32(f, (std::string("tidal.") + field.name).c_str(), c.*(field.value));
    gguf_set_val_f32(f, "tidal.eps", c.eps);
    gguf_set_val_f32(f, "tidal.rope_base", c.rope_base);
    if (c.experts) {
        gguf_set_val_u32(f, "tidal.experts", c.experts);
        gguf_set_val_u32(f, "tidal.experts_used", c.experts_used);
        gguf_set_val_u32(f, "tidal.expert_ffn", c.expert_ffn);
        gguf_set_val_u32(f, "tidal.shared_ffn", c.shared_ffn);
        gguf_set_val_bool(f, "tidal.norm_topk", c.norm_topk);
    }
    write_ple(f, c.ple);
}

Config read_config(const gguf_context * f) {
    const std::string arch = gguf_get_val_str(f, key(f, "general.architecture", GGUF_TYPE_STRING));
    if (arch != architecture && arch != moe_architecture)
        throw std::runtime_error("Expected tidal-qwen4exp-dense or tidal-qwen4exp-moe; convert external checkpoints first");
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
    const IntField moe_fields[] = {{"experts", &Config::experts}, {"experts_used", &Config::experts_used},
                                  {"expert_ffn", &Config::expert_ffn}, {"shared_ffn", &Config::shared_ffn}};
    for (auto field : moe_fields) {
        const auto name = std::string("tidal.") + field.name;
        if (arch == moe_architecture) {
            auto value = gguf_get_val_u32(f, key(f, name, GGUF_TYPE_UINT32));
            if (value > uint32_t(INT32_MAX)) throw std::runtime_error("MoE metadata overflow");
            c.*(field.value) = int(value);
        } else if (gguf_find_key(f, name.c_str()) >= 0) throw std::runtime_error("MoE metadata in Dense architecture");
    }
    if (arch == moe_architecture) {
        c.norm_topk = gguf_get_val_bool(f, key(f, "tidal.norm_topk", GGUF_TYPE_BOOL));
        if (!c.experts) throw std::runtime_error("MoE model must have experts");
    } else if (gguf_find_key(f, "tidal.norm_topk") >= 0) throw std::runtime_error("MoE metadata in Dense architecture");
    read_ple(f, c);
    c.validate();
    return c;
}

Model::Model(const std::string & path, ggml_backend_t backend) {
    ggml_context * raw = nullptr;
    Gguf file(gguf_init_from_file(path.c_str(), {true, &raw}), gguf_free);
    Context source(raw, ggml_free);
    if (!file || !source) throw std::runtime_error("Cannot read GGUF: " + path);
    config = read_config(file.get());
    const auto specs = tensor_specs(config);
    if (gguf_get_n_tensors(file.get()) != int64_t(specs.size()))
        throw std::runtime_error("Unexpected number of model tensors");
    std::ifstream input(path, std::ios::binary);
    input.seekg(0, std::ios::end);
    const auto length = input.tellg();
    if (length < 0) throw std::runtime_error("Cannot determine model file length");
    ctx.reset(ggml_init({specs.size() * ggml_tensor_overhead() + 4096, nullptr, true}));
    if (!ctx) throw std::runtime_error("Cannot allocate weight metadata");
    for (const auto & spec : specs) {
        auto * t = ggml_get_tensor(source.get(), spec.name.c_str());
        if (!t || t->ne[0] != spec.columns || t->ne[1] != spec.rows || t->ne[2] != spec.slices || t->ne[3] != 1)
            throw std::runtime_error("Missing or invalid tensor shape: " + spec.name);
        if (!storage_type(t->type))
            throw std::runtime_error("Unsupported tensor type: " + spec.name);
        const auto id = gguf_find_tensor(file.get(), spec.name.c_str());
        const uint64_t offset = gguf_get_data_offset(file.get()) + gguf_get_tensor_offset(file.get(), id);
        if (offset > uint64_t(length) || ggml_nbytes(t) > uint64_t(length) - offset)
            throw std::runtime_error("Truncated GGUF tensor: " + spec.name);
        auto type = t->type;
        const bool lookup = spec.kind == TensorSpec::Lookup || spec.kind == TensorSpec::Embedding;
        const bool expert = spec.kind == TensorSpec::Expert;
        const bool matrix = expert || spec.kind == TensorSpec::Matrix || spec.kind == TensorSpec::Router;
        if (!matrix && !lookup) type = GGML_TYPE_F32;
        else {
            Context probe(ggml_init({65536, nullptr, true}), ggml_free);
            if (!probe) throw std::runtime_error("Cannot allocate backend probe");
            auto * x = ggml_new_tensor_1d(probe.get(), lookup ? GGML_TYPE_I32 : GGML_TYPE_F32,
                                           lookup ? 1 : spec.columns);
            auto * op = lookup ? ggml_get_rows(probe.get(), t, x) : expert
                ? ggml_mul_mat_id(probe.get(), t, x, ggml_new_tensor_2d(probe.get(), GGML_TYPE_I32, 1, 1))
                : ggml_mul_mat(probe.get(), t, x);
            if (matrix && !expert) ggml_prec_set_acc(op, GGML_PREC_F32);
            const bool cpu_dot = !matrix || !ggml_backend_is_cpu(backend) || ggml_get_type_traits_cpu(type)->vec_dot;
            if (!cpu_dot || !ggml_backend_supports_op(backend, op)) type = GGML_TYPE_F32;
        }
        auto * tensor = ggml_new_tensor_3d(ctx.get(), type, spec.columns, spec.rows, spec.slices);
        ggml_set_name(tensor, spec.name.c_str());
        weights.emplace(spec.name, tensor);
        info.parameters += uint64_t(ggml_nelements(t));
        info.stored_weight_bytes += ggml_nbytes(t);
        info.resident_weight_bytes += ggml_nbytes(tensor);
        if (type != t->type) {
            ++info.converted_tensors;
            std::cerr << "weight conversion: " << spec.name << ' ' << ggml_type_name(t->type)
                      << " -> " << ggml_type_name(type) << " on " << ggml_backend_name(backend) << '\n';
        }
    }
    std::cerr << "model weight bytes: stored=" << info.stored_weight_bytes << " resident="
              << info.resident_weight_bytes << " converted_tensors=" << info.converted_tensors << '\n';
    buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (!buffer) throw std::runtime_error("Cannot allocate model weights");
    info.weight_buffer_bytes = ggml_backend_buffer_get_size(buffer.get());
    ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    // Read in file order with bounded staging memory, including converted tensors.
    for (int64_t id = 0; id < gguf_get_n_tensors(file.get()); ++id) {
        const std::string name = gguf_get_tensor_name(file.get(), id);
        auto * tensor = weights.at(name);
        auto * encoded = ggml_get_tensor(source.get(), name.c_str());
        const uint64_t offset = gguf_get_data_offset(file.get()) + gguf_get_tensor_offset(file.get(), id);
        const size_t row_bytes = ggml_row_size(encoded->type, encoded->ne[0]);
        const int64_t rows = ggml_nelements(encoded) / encoded->ne[0];
        const int64_t chunk_rows = std::max<int64_t>(1, (8 * 1024 * 1024) / (encoded->ne[0] * sizeof(float)));
        std::vector<char> data(size_t(std::min(rows, chunk_rows)) * row_bytes);
        std::vector<float> converted(encoded->type == tensor->type ? 0 : size_t(std::min(rows, chunk_rows) * encoded->ne[0]));
        input.seekg(std::streamoff(offset));
        for (int64_t row = 0; row < rows; row += chunk_rows) {
            const int64_t count = std::min(chunk_rows, rows - row);
            const size_t bytes = size_t(count) * row_bytes;
            input.read(data.data(), std::streamsize(bytes));
            if (!input) throw std::runtime_error("Failed to read tensor: " + name);
            if (converted.empty()) ggml_backend_tensor_set(tensor, data.data(), size_t(row) * row_bytes, bytes);
            else {
                dequantize(encoded->type, data.data(), converted.data(), count * encoded->ne[0]);
                ggml_backend_tensor_set(tensor, converted.data(), size_t(row * encoded->ne[0]) * sizeof(float),
                                          size_t(count * encoded->ne[0]) * sizeof(float));
            }
        }
    }
}
} // namespace tidal
