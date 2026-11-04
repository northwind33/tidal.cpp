#include "internal.hpp"
#include "args.hpp"

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

int main(int argc, char ** argv) {
    try {
        Args args(argc, argv, {"--help", "--output", "--type", "--seed", "--context", "--layers", "--hidden"});
        if (!args.get("--help").empty()) {
            std::cout << "tidal-dummy --output model.gguf [--type f32|f16|bf16|q8_0|q4_0] "
                         "[--seed 42] [--context 256] [--layers 4] [--hidden 128]\n";
            return 0;
        }
        const auto path = args.get("--output");
        if (path.empty()) throw std::invalid_argument("--output is required");
        if (std::filesystem::exists(path)) throw std::invalid_argument("Refusing to overwrite an existing model: " + path);
        const auto name = args.get("--type", "f32");
        const std::map<std::string, ggml_type> types = {{"f32", GGML_TYPE_F32}, {"f16", GGML_TYPE_F16},
            {"bf16", GGML_TYPE_BF16}, {"q8_0", GGML_TYPE_Q8_0}, {"q4_0", GGML_TYPE_Q4_0}};
        if (!types.contains(name)) throw std::invalid_argument("Unsupported weight type: " + name);
        const int seed = args.integer("--seed", 42);
        if (seed <= 0) throw std::invalid_argument("Seed must be a positive 32-bit integer");
        Random rng{uint32_t(seed)};
        tidal::Config config;
        config.context = args.integer("--context", config.context);
        config.layers = args.integer("--layers", config.layers);
        config.hidden = args.integer("--hidden", config.hidden);
        if (config.hidden <= 0 || config.hidden > 1024) throw std::invalid_argument("hidden must be in [32, 1024]");
        config.ffn = 2 * config.hidden;
        config.validate();
        tidal::Context ctx(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
        tidal::Gguf file(gguf_init_empty(), gguf_free);
        if (!ctx || !file) throw std::runtime_error("Cannot allocate GGUF writer");
        tidal::write_config(file.get(), config);
        gguf_set_val_u32(file.get(), "tidal.seed", uint32_t(seed));
        gguf_set_val_str(file.get(), "tidal.weight_type", name.c_str());
        gguf_set_val_str(file.get(), "general.description", "Random tiny Dense derivative; not trained, no tokenizer, no MoE/PLE/MTP");
        std::vector<std::vector<uint8_t>> storage;
        size_t parameters = 0, bytes = 0;
        for (const auto & spec : tidal::tensor_specs(config)) {
            ggml_type type = GGML_TYPE_F32;
            if (spec.kind == tidal::TensorSpec::Matrix) type = types.at(name);
            if (spec.kind == tidal::TensorSpec::Embedding) type = GGML_TYPE_BF16;
            if (spec.columns % ggml_blck_size(type)) throw std::runtime_error("Tensor width is not quantization aligned");
            auto * tensor = ggml_new_tensor_2d(ctx.get(), type, spec.columns, spec.rows);
            ggml_set_name(tensor, spec.name.c_str());
            const auto count = size_t(spec.columns * spec.rows);
            std::vector<float> source(count);
            for (float & value : source) {
                float r = 2 * rng.uniform() - 1;
                switch (spec.kind) {
                    case tidal::TensorSpec::Norm: value = 1 + 0.05f * r; break; // already folded (1+w)
                    case tidal::TensorSpec::Conv: value = 0.3f * r; break;
                    case tidal::TensorSpec::ALog: value = -0.5f + 0.2f * r; break;
                    case tidal::TensorSpec::Bias: value = -1.0f + 0.2f * r; break;
                    case tidal::TensorSpec::Embedding: value = 0.1f * r; break;
                    default: value = r * std::sqrt(3.0f / float(spec.columns)); break;
                }
            }
            storage.emplace_back(ggml_nbytes(tensor));
            auto & encoded = storage.back();
            const auto written = ggml_quantize_chunk(type, source.data(), encoded.data(), 0, spec.rows, spec.columns, nullptr);
            if (written != encoded.size()) throw std::runtime_error("Unexpected quantized tensor length");
            tensor->data = encoded.data();
            gguf_add_tensor(file.get(), tensor);
            bytes += encoded.size(); parameters += count;
        }
        if (!gguf_write_to_file(file.get(), path.c_str(), false)) throw std::runtime_error("Failed to write " + path);
        std::cout << path << ": " << parameters << " parameters, " << bytes << " tensor bytes, " << name << " matrices\n";
        return 0;
    } catch (const std::exception & e) { std::cerr << "tidal-dummy: " << e.what() << '\n'; return 1; }
}
