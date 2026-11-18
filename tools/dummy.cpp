#include "internal.hpp"
#include "args.hpp"
#include "weight_writer.hpp"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>

// A specified integer RNG, rather than std::normal_distribution, makes the exact
// source weights reproducible across standard-library implementations.
struct Random {
    uint32_t state;
    float uniform() {
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        return float(state >> 8) / 16777216.0f;
    }
};

// Qwen's deterministic SplitMix64 hash constants and successive prime head
// vocabularies. The GGUF stores the resulting constants, not a platform RNG seed.
uint64_t splitmix(uint64_t value) {
    value += UINT64_C(0x9E3779B97F4A7C15);
    value = (value ^ (value >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94D049BB133111EB);
    return value ^ (value >> 31);
}
bool prime(int value) {
    if (value < 2) return false;
    for (int d = 2; d <= value / d; ++d) if (value % d == 0) return false;
    return true;
}
void configure_ple(tidal::Config & c, const Args & args, int seed) {
    const std::vector<std::string> options = {"--ple-ngram", "--ple-heads", "--ple-dim",
        "--ple-conv", "--ple-eos", "--ple-vocab-base", "--ple-type"};
    if (!args.contains("--ple-layers")) {
        for (const auto & option : options)
            if (args.contains(option)) throw std::invalid_argument(option + " requires --ple-layers");
        return;
    }
    auto & p = c.ple;
    p.ngram_size = args.integer("--ple-ngram", p.ngram_size);
    p.heads_per_ngram = args.integer("--ple-heads", p.heads_per_ngram);
    p.embedding_dim = args.integer("--ple-dim", p.embedding_dim);
    p.conv_kernel = args.integer("--ple-conv", p.conv_kernel);
    p.eos_token_id = args.integer("--ple-eos", p.eos_token_id);
    const int base = args.integer("--ple-vocab-base", 17);
    if (p.ngram_size < 2 || p.ngram_size > 16 || p.heads_per_ngram < 1 || p.heads_per_ngram > 64 ||
        p.heads() > 64 || base < 2 || base > 4096)
        throw std::invalid_argument("Invalid dummy PLE hash dimensions or vocabulary base");
    std::string ids = args.get("--ple-layers");
    int next_prime = base - 1;
    for (size_t start = 0;;) {
        auto end = ids.find(',', start);
        tidal::PLELayer l;
        l.layer = number(ids.substr(start, end == std::string::npos ? end : end - start));
        if (l.layer < 0 || l.layer >= c.layers || (l.layer + 1) % 4 == 0 ||
            (!p.layers.empty() && l.layer <= p.layers.back().layer))
            throw std::invalid_argument("PLE layers must be sorted, unique, zero-based GDN layers");
        const uint64_t bound = (uint64_t(INT64_MAX) / uint64_t(c.vocab)) / 2;
        for (int j = 0; j < p.ngram_size; ++j)
            l.multipliers.push_back(2 * (splitmix(uint64_t(seed) + 10007 * p.layers.size() +
                UINT64_C(0x9E3779B97F4A7C15) * (j + 1)) % bound) + 1);
        for (int h = 0; h < p.heads(); ++h) {
            do { ++next_prime; } while (!prime(next_prime));
            l.offsets.push_back(uint64_t(l.vocab));
            l.vocab_sizes.push_back(uint64_t(next_prime));
            l.vocab += next_prime;
        }
        l.vocab = (l.vocab + 127) / 128 * 128;
        p.layers.push_back(std::move(l));
        if (end == std::string::npos) break;
        start = end + 1;
    }
    c.validate();
}

// A random vision tower in the Qwen3-VL mmproj layout (clip.projector_type
// qwen3vl_merger), shaped like the released towers but small.
void write_vision(const Args & args, const std::string & path, ggml_type type, uint32_t seed) {
    const int hidden = args.integer("--vision-hidden", 128), heads = args.integer("--vision-heads", 2);
    const int layers = args.integer("--vision-layers", 2), ffn = args.integer("--vision-ffn", 256);
    const int output = args.integer("--vision-output", 128), side = args.integer("--vision-position-side", 8);
    const int patch = 16;
    if (hidden < 32 || hidden > 4096 || heads < 1 || hidden % heads || (hidden / heads) % 4 || layers < 1 || layers > 64 ||
        ffn < 32 || output < 32 || side < 2 || side > 64)
        throw std::invalid_argument("Invalid dummy vision dimensions");
    struct Spec { std::string name; std::vector<int64_t> shape; int kind; }; // 0 matrix, 1 bias, 2 norm, 3 patch, 4 position
    std::vector<Spec> specs = {{"v.patch_embd.weight", {patch, patch, 3, hidden}, 3}, {"v.patch_embd.weight.1", {patch, patch, 3, hidden}, 3},
                               {"v.patch_embd.bias", {hidden}, 1}, {"v.position_embd.weight", {hidden, int64_t(side) * side}, 4}};
    for (int l = 0; l < layers; ++l) {
        const auto p = "v.blk." + std::to_string(l) + ".";
        specs.push_back({p + "ln1.weight", {hidden}, 2}); specs.push_back({p + "ln1.bias", {hidden}, 1});
        specs.push_back({p + "attn_qkv.weight", {hidden, 3 * hidden}, 0}); specs.push_back({p + "attn_qkv.bias", {3 * hidden}, 1});
        specs.push_back({p + "attn_out.weight", {hidden, hidden}, 0}); specs.push_back({p + "attn_out.bias", {hidden}, 1});
        specs.push_back({p + "ln2.weight", {hidden}, 2}); specs.push_back({p + "ln2.bias", {hidden}, 1});
        specs.push_back({p + "ffn_up.weight", {hidden, ffn}, 0}); specs.push_back({p + "ffn_up.bias", {ffn}, 1});
        specs.push_back({p + "ffn_down.weight", {ffn, hidden}, 0}); specs.push_back({p + "ffn_down.bias", {hidden}, 1});
    }
    specs.push_back({"v.post_ln.weight", {hidden}, 2}); specs.push_back({"v.post_ln.bias", {hidden}, 1});
    specs.push_back({"mm.0.weight", {4 * hidden, 4 * hidden}, 0}); specs.push_back({"mm.0.bias", {4 * hidden}, 1});
    specs.push_back({"mm.2.weight", {4 * hidden, output}, 0}); specs.push_back({"mm.2.bias", {output}, 1});
    size_t bytes = 0;
    for (const auto & s : specs) {
        int64_t n = 1; for (auto d : s.shape) n *= d;
        const auto t = s.kind == 0 ? type : GGML_TYPE_F32;
        if (s.shape[0] % ggml_blck_size(t)) throw std::invalid_argument(s.name + ": row width not divisible by the block size");
        bytes += ggml_row_size(t, n) + 256;
    }
    tidal::Context ctx(ggml_init({bytes + specs.size() * ggml_tensor_overhead() + 4096, nullptr, false}), ggml_free);
    tidal::Gguf file(gguf_init_empty(), gguf_free);
    if (!ctx || !file) throw std::runtime_error("Cannot allocate the vision GGUF writer");
    auto * f = file.get();
    gguf_set_val_str(f, "general.architecture", "clip");
    gguf_set_val_str(f, "general.type", "mmproj");
    gguf_set_val_str(f, "general.description", "Random synthetic Qwen3-VL vision tower; not trained");
    gguf_set_val_bool(f, "clip.has_vision_encoder", true);
    gguf_set_val_str(f, "clip.projector_type", "qwen3vl_merger");
    gguf_set_val_bool(f, "clip.use_gelu", true);
    gguf_set_val_u32(f, "clip.vision.image_size", uint32_t(side * patch));
    gguf_set_val_u32(f, "clip.vision.patch_size", uint32_t(patch));
    gguf_set_val_u32(f, "clip.vision.embedding_length", uint32_t(hidden));
    gguf_set_val_u32(f, "clip.vision.feed_forward_length", uint32_t(ffn));
    gguf_set_val_u32(f, "clip.vision.block_count", uint32_t(layers));
    gguf_set_val_u32(f, "clip.vision.attention.head_count", uint32_t(heads));
    gguf_set_val_u32(f, "clip.vision.projection_dim", uint32_t(output));
    gguf_set_val_u32(f, "clip.vision.spatial_merge_size", 2);
    gguf_set_val_f32(f, "clip.vision.attention.layer_norm_epsilon", 1e-6f);
    const float mean[3] = {0.5f, 0.5f, 0.5f}, std_[3] = {0.5f, 0.5f, 0.5f};
    gguf_set_arr_data(f, "clip.vision.image_mean", GGUF_TYPE_FLOAT32, mean, 3);
    gguf_set_arr_data(f, "clip.vision.image_std", GGUF_TYPE_FLOAT32, std_, 3);
    Random rng{seed};
    std::vector<float> values;
    for (const auto & s : specs) {
        const auto t = s.kind == 0 ? type : GGML_TYPE_F32;
        int64_t n = 1; for (auto d : s.shape) n *= d;
        values.resize(size_t(n));
        const float fan_in = float(s.kind == 3 ? patch * patch * 3 : s.shape[0]);
        for (auto & v : values) {
            const float r = 2 * rng.uniform() - 1;
            v = s.kind == 0 || s.kind == 3 ? r * std::sqrt(3.0f / fan_in) : s.kind == 1 ? 0.02f * r : s.kind == 2 ? 1 + 0.05f * r : 0.1f * r;
        }
        ggml_tensor * tensor = nullptr;
        if (s.shape.size() == 1) tensor = ggml_new_tensor_1d(ctx.get(), t, s.shape[0]);
        else if (s.shape.size() == 2) tensor = ggml_new_tensor_2d(ctx.get(), t, s.shape[0], s.shape[1]);
        else tensor = ggml_new_tensor_4d(ctx.get(), t, s.shape[0], s.shape[1], s.shape[2], s.shape[3]);
        ggml_set_name(tensor, s.name.c_str());
        if (t == GGML_TYPE_F32) std::memcpy(tensor->data, values.data(), values.size() * sizeof(float));
        else ggml_quantize_chunk(t, values.data(), tensor->data, 0, n / s.shape[0], s.shape[0], nullptr);
        gguf_add_tensor(f, tensor);
    }
    if (!gguf_write_to_file(f, path.c_str(), false)) throw std::runtime_error("Cannot write " + path);
    std::cout << "vision tower: hidden=" << hidden << " heads=" << heads << " layers=" << layers << " output=" << output << '\n';
}

int main(int argc, char ** argv) {
    try {
        Args args(argc, argv, {"--help", "--output", "--type", "--seed", "--context", "--layers", "--hidden",
            "--vision", "--vision-hidden", "--vision-heads", "--vision-layers", "--vision-ffn", "--vision-output",
            "--vision-position-side", "--image-token",
            "--ple-layers", "--ple-ngram", "--ple-heads", "--ple-dim", "--ple-conv", "--ple-eos",
            "--ple-vocab-base", "--ple-type", "--experts", "--experts-used", "--expert-ffn", "--shared-ffn", "--norm-topk",
            "--list-types", "--tensor-types", "--dry-run", "--threads", "--chunk-mib", "--preset", "--rank", "--ffn", "--vocab"});
        if (!args.get("--help").empty()) {
            std::cout << "tidal-dummy --output model.gguf [--type FORMAT] "
                         "[--seed 42] [--context 256] [--layers 4] [--hidden 128]\n"
                         "  Optional PLE: --ple-layers 0,2 (zero-based GDN layers)\n"
                         "  [--ple-ngram 3] [--ple-heads 2] [--ple-dim 128] [--ple-conv 4]\n"
                         "  [--ple-eos 0] [--ple-vocab-base 17] [--ple-type bf16]\n";
            std::cout << "  Optional MoE: --experts 4 [--experts-used 2] [--expert-ffn 128] [--shared-ffn 128] [--norm-topk 1]\n";
            std::cout << "  --list-types 1 | --preset tiny|quant-test|dense-27b | --tensor-types rules.txt (glob=type, last match wins)\n"
                         "  --dry-run 1 plans without writing; --chunk-mib 16 --threads 4 bound generation memory\n"
                         "  --image-token ID records the image placeholder token of the text model.\n"
                         "  --vision 1 writes a random Qwen3-VL vision tower (mmproj) instead: [--vision-hidden 128]\n"
                         "  [--vision-heads 2] [--vision-layers 2] [--vision-ffn 256] [--vision-output 128 (text hidden)]\n"
                         "  [--vision-position-side 8]; --type sets its matrix format.\n";
            return 0;
        }
        if (args.integer("--list-types", 0)) {
            for (auto type : tidal::storage_types())
                std::cout << ggml_type_name(type) << " block=" << ggml_blck_size(type)
                          << " bpw=" << 8.0 * double(ggml_type_size(type)) / double(ggml_blck_size(type))
                          << " requires_imatrix=" << ggml_quantize_requires_imatrix(type) << '\n';
            return 0;
        }
        const auto path = args.get("--output");
        const int dry = args.integer("--dry-run", 0);
        const int threads = args.integer("--threads", 4), chunk_mib = args.integer("--chunk-mib", 16);
        if (dry < 0 || dry > 1 || threads < 1 || threads > 64 || chunk_mib < 1 || chunk_mib > 256)
            throw std::invalid_argument("Invalid dry-run, thread count or chunk size");
        if (!dry && path.empty()) throw std::invalid_argument("--output is required");
        if (!dry && std::filesystem::exists(path)) throw std::invalid_argument("Refusing to overwrite an existing model: " + path);
        const auto name = args.get("--type", "f32");
        const auto type = tidal::parse_storage_type(name);
        const int seed = args.integer("--seed", 42);
        if (seed <= 0) throw std::invalid_argument("Seed must be a positive 32-bit integer");
        if (args.integer("--vision", 0)) { write_vision(args, path, type, uint32_t(seed)); return 0; }
        Random rng{uint32_t(seed)};
        tidal::Config config;
        const auto preset = args.get("--preset", "tiny");
        if (preset == "quant-test") { config.hidden = 256; config.rank = 256; config.gdn_dim = 64; config.head_dim = 64; }
        else if (preset == "dense-27b") {
            config.hidden = 5120; config.ffn = 18944; config.vocab = 131072;
            config.layers = 64; config.context = 2048; config.rank = 256;
            config.q_heads = 40; config.kv_heads = 8; config.head_dim = 128; config.rope_dim = 64;
            config.gdn_key_heads = 8; config.gdn_value_heads = 40; config.gdn_dim = 128;
            config.index_dim = 128; config.block_size = 16; config.top_blocks = 8;
        }
        else if (preset != "tiny") throw std::invalid_argument("Unknown model preset: " + preset);
        config.context = args.integer("--context", config.context);
        config.layers = args.integer("--layers", config.layers);
        config.hidden = args.integer("--hidden", config.hidden);
        config.rank = args.integer("--rank", config.rank);
        if (config.hidden <= 0 || config.hidden > 16384) throw std::invalid_argument("hidden must be in [32, 16384]");
        config.ffn = args.integer("--ffn", preset == "dense-27b" ? config.ffn : 2 * config.hidden);
        config.vocab = args.integer("--vocab", config.vocab);
        config.experts = args.integer("--experts", 0);
        config.experts_used = args.integer("--experts-used", config.experts ? 2 : 0);
        config.expert_ffn = args.integer("--expert-ffn", config.experts ? config.hidden : 0);
        config.shared_ffn = args.integer("--shared-ffn", config.experts ? config.hidden : 0);
        const int norm_topk = args.integer("--norm-topk", 1);
        if (norm_topk != 0 && norm_topk != 1) throw std::invalid_argument("--norm-topk must be 0 or 1");
        config.norm_topk = bool(norm_topk);
        config.image_token = args.integer("--image-token", -1);
        if (config.image_token < -1 || config.image_token >= config.vocab) throw std::invalid_argument("--image-token outside the vocabulary");
        if (config.image_token >= 0) tidal::default_rope_sections(config);
        config.validate();
        configure_ple(config, args, seed);
        const auto ple_type = tidal::parse_storage_type(args.get("--ple-type", "bf16"));
        const size_t context_bytes = tidal::tensor_specs(config).size() * ggml_tensor_overhead() + 4096;
        tidal::Context ctx(ggml_init({context_bytes, nullptr, true}), ggml_free);
        tidal::Gguf file(gguf_init_empty(), gguf_free);
        if (!ctx || !file) throw std::runtime_error("Cannot allocate GGUF writer");
        tidal::write_config(file.get(), config);
        gguf_set_val_u32(file.get(), "tidal.seed", uint32_t(seed));
        gguf_set_val_str(file.get(), "tidal.weight_type", name.c_str());
        // Deep random residual networks amplify tiny backend rounding changes.
        // Scale residual output projections at initialization, without changing
        // the model equations or the legacy tiny fixtures.
        const float residual_scale = preset == "dense-27b" ? 1.0f / float(config.layers) : 1.0f;
        if (preset == "dense-27b") gguf_set_val_f32(file.get(), "tidal.residual_init_scale", residual_scale);
        gguf_set_val_str(file.get(), "general.description", config.experts ? "Random synthetic Qwen-derived MoE; not trained, no tokenizer/MTP" :
            preset == "dense-27b" ? "Random synthetic large Qwen-derived Dense; not trained, no tokenizer/MTP" : config.ple.layers.empty()
            ? "Random tiny Dense derivative; not trained, no tokenizer, no MoE/PLE/MTP"
            : "Random tiny Dense derivative with PLE; not trained, no tokenizer, no MoE/MTP");
        auto rules = read_type_rules(args.get("--tensor-types"));
        const auto plan = plan_weights(config, ctx.get(), file.get(), type, ple_type, rules);
        if (dry) return 0;
        write_weights(path, file.get(), plan, chunk_mib, threads, [&](const tidal::TensorSpec & spec, float * data, int64_t count) {
            float scale = std::sqrt(3.0f / float(spec.columns));
            if (spec.name.ends_with(".ffn.down") || spec.name.ends_with(".gdn.out") || spec.name.ends_with(".attn.out"))
                scale *= residual_scale;
            for (int64_t i = 0; i < count; ++i) {
                float & value = data[i];
                float r = 2 * rng.uniform() - 1;
                switch (spec.kind) {
                    case tidal::TensorSpec::Norm: value = 1 + 0.05f * r; break; // already folded (1+w)
                    case tidal::TensorSpec::Conv: value = 0.3f * r; break;
                    case tidal::TensorSpec::ALog: value = -0.5f + 0.2f * r; break;
                    case tidal::TensorSpec::Bias: value = -1.0f + 0.2f * r; break;
                    case tidal::TensorSpec::Embedding:
                    case tidal::TensorSpec::Lookup: value = 0.1f * r; break;
                    default: value = r * scale; break;
                }
            }
        });
        return 0;
    } catch (const std::exception & e) { std::cerr << "tidal-dummy: " << e.what() << '\n'; return 1; }
}
