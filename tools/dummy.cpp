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

int main(int argc, char ** argv) {
    try {
        Args args(argc, argv, {"--help", "--output", "--type", "--seed", "--context", "--layers", "--hidden",
            "--ple-layers", "--ple-ngram", "--ple-heads", "--ple-dim", "--ple-conv", "--ple-eos",
            "--ple-vocab-base", "--ple-type", "--experts", "--experts-used", "--expert-ffn", "--shared-ffn", "--norm-topk",
            "--list-types", "--tensor-types", "--dry-run", "--threads", "--chunk-mib", "--preset", "--rank"});
        if (!args.get("--help").empty()) {
            std::cout << "tidal-dummy --output model.gguf [--type FORMAT] "
                         "[--seed 42] [--context 256] [--layers 4] [--hidden 128]\n"
                         "  Optional PLE: --ple-layers 0,2 (zero-based GDN layers)\n"
                         "  [--ple-ngram 3] [--ple-heads 2] [--ple-dim 128] [--ple-conv 4]\n"
                         "  [--ple-eos 0] [--ple-vocab-base 17] [--ple-type bf16]\n";
            std::cout << "  Optional MoE: --experts 4 [--experts-used 2] [--expert-ffn 128] [--shared-ffn 128] [--norm-topk 1]\n";
            std::cout << "  --list-types 1 | --preset quant-test | --tensor-types rules.txt (glob=type, last match wins)\n"
                         "  --dry-run 1 plans without writing; --chunk-mib 16 --threads 4 bound generation memory\n";
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
        Random rng{uint32_t(seed)};
        tidal::Config config;
        const auto preset = args.get("--preset", "tiny");
        if (preset == "quant-test") { config.hidden = 256; config.rank = 256; config.gdn_dim = 64; config.head_dim = 64; }
        else if (preset != "tiny") throw std::invalid_argument("Unknown model preset: " + preset);
        config.context = args.integer("--context", config.context);
        config.layers = args.integer("--layers", config.layers);
        config.hidden = args.integer("--hidden", config.hidden);
        config.rank = args.integer("--rank", config.rank);
        if (config.hidden <= 0 || config.hidden > 1024) throw std::invalid_argument("hidden must be in [32, 1024]");
        config.ffn = 2 * config.hidden;
        config.experts = args.integer("--experts", 0);
        config.experts_used = args.integer("--experts-used", config.experts ? 2 : 0);
        config.expert_ffn = args.integer("--expert-ffn", config.experts ? config.hidden : 0);
        config.shared_ffn = args.integer("--shared-ffn", config.experts ? config.hidden : 0);
        const int norm_topk = args.integer("--norm-topk", 1);
        if (norm_topk != 0 && norm_topk != 1) throw std::invalid_argument("--norm-topk must be 0 or 1");
        config.norm_topk = bool(norm_topk);
        config.validate();
        configure_ple(config, args, seed);
        const auto ple_type = tidal::parse_storage_type(args.get("--ple-type", "bf16"));
        tidal::Context ctx(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
        tidal::Gguf file(gguf_init_empty(), gguf_free);
        if (!ctx || !file) throw std::runtime_error("Cannot allocate GGUF writer");
        tidal::write_config(file.get(), config);
        gguf_set_val_u32(file.get(), "tidal.seed", uint32_t(seed));
        gguf_set_val_str(file.get(), "tidal.weight_type", name.c_str());
        gguf_set_val_str(file.get(), "general.description", config.experts ? "Random synthetic Qwen-derived MoE; not trained, no tokenizer/MTP" : config.ple.layers.empty()
            ? "Random tiny Dense derivative; not trained, no tokenizer, no MoE/PLE/MTP"
            : "Random tiny Dense derivative with PLE; not trained, no tokenizer, no MoE/MTP");
        auto rules = read_type_rules(args.get("--tensor-types"));
        const auto plan = plan_weights(config, ctx.get(), file.get(), type, ple_type, rules);
        if (dry) return 0;
        write_weights(path, file.get(), plan, chunk_mib, threads, [&](const tidal::TensorSpec & spec, float * data, int64_t count) {
            const float scale = std::sqrt(3.0f / float(spec.columns));
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
