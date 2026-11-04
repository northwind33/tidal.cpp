#include "internal.hpp"
#include "ggml-cpu.h"
#ifdef TIDAL_CUDA
#include "ggml-cuda.h"
#endif
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>

int main(int argc, char ** argv) {
    try {
        if (argc != 3) throw std::runtime_error("Usage: tidal-weights-test model.gguf cpu|cuda");
        tidal::Backend backend(nullptr, ggml_backend_free);
        if (std::string(argv[2]) == "cpu") {
            backend.reset(ggml_backend_cpu_init());
            ggml_backend_cpu_set_n_threads(backend.get(), 2);
        } else {
#ifdef TIDAL_CUDA
            backend.reset(ggml_backend_cuda_init(0));
#endif
        }
        if (!backend) throw std::runtime_error("Requested backend unavailable");
        tidal::Model model(argv[1], backend.get());
        ggml_context * raw = nullptr;
        tidal::Gguf file(gguf_init_from_file(argv[1], {true, &raw}), gguf_free);
        tidal::Context metadata(raw, ggml_free);
        std::ifstream input(argv[1], std::ios::binary);
        int verified = 0;
        for (const auto & spec : tidal::tensor_specs(model.config)) {
            auto * weight = model.at(spec.name);
            auto * stored = ggml_get_tensor(metadata.get(), spec.name.c_str());
            const auto id = gguf_find_tensor(file.get(), spec.name.c_str());
            std::vector<char> encoded(ggml_nbytes(stored)), resident(ggml_nbytes(weight));
            input.seekg(gguf_get_data_offset(file.get()) + gguf_get_tensor_offset(file.get(), id));
            input.read(encoded.data(), encoded.size());
            if (!input) throw std::runtime_error("Cannot read reference weights");
            ggml_backend_tensor_get(weight, resident.data(), 0, resident.size());
            std::vector<float> decoded(size_t(ggml_nelements(stored)));
            tidal::dequantize(stored->type, encoded.data(), decoded.data(), ggml_nelements(stored));
            if (stored->type == weight->type) {
                if (encoded != resident) throw std::runtime_error("Uploaded bytes differ: " + spec.name);
            } else if (weight->type != GGML_TYPE_F32 ||
                       std::memcmp(decoded.data(), resident.data(), resident.size()))
                throw std::runtime_error("Load-time conversion differs: " + spec.name);
            const bool lookup = spec.kind == tidal::TensorSpec::Embedding || spec.kind == tidal::TensorSpec::Lookup;
            const bool matrix = spec.kind == tidal::TensorSpec::Matrix || spec.kind == tidal::TensorSpec::Router ||
                                spec.kind == tidal::TensorSpec::Expert;
            if (!lookup && !matrix) continue;
            const int32_t expert = int32_t(spec.slices - 1), row = int32_t(spec.rows - 1);
            tidal::Context ctx(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
            auto * x = ggml_new_tensor_1d(ctx.get(), lookup ? GGML_TYPE_I32 : GGML_TYPE_F32, lookup ? 1 : spec.columns);
            auto * ids = spec.kind == tidal::TensorSpec::Expert ? ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 1, 1) : nullptr;
            auto * y = lookup ? ggml_get_rows(ctx.get(), weight, x) : ids ? ggml_mul_mat_id(ctx.get(), weight, x, ids)
                                                                                       : ggml_mul_mat(ctx.get(), weight, x);
            if (matrix && !ids) ggml_prec_set_acc(y, GGML_PREC_F32);
            if (!ggml_backend_supports_op(backend.get(), y)) throw std::runtime_error("Native op unsupported: " + spec.name);
            auto * graph = ggml_new_graph(ctx.get());
            ggml_build_forward_expand(graph, y);
            tidal::Buffer buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()), ggml_backend_buffer_free);
            if (!buffer) throw std::runtime_error("Test graph allocation failed");
            std::vector<float> values(size_t(spec.columns));
            for (int64_t i = 0; i < spec.columns; ++i) values[i] = std::sin(float(i + 1) * 0.731f);
            if (lookup) ggml_backend_tensor_set(x, &row, 0, sizeof(row));
            else ggml_backend_tensor_set(x, values.data(), 0, values.size() * sizeof(float));
            if (ids) ggml_backend_tensor_set(ids, &expert, 0, sizeof(expert));
            if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("Weight operation failed");
            std::vector<float> result(size_t(ggml_nelements(y)));
            ggml_backend_tensor_get(y, result.data(), 0, result.size() * sizeof(float));
            const bool rounded_inputs = ggml_backend_is_cpu(backend.get()) || ids;
            for (float & value : values) {
                if (rounded_inputs && weight->type == GGML_TYPE_F16) value = ggml_fp16_to_fp32(ggml_fp32_to_fp16(value));
                if (rounded_inputs && weight->type == GGML_TYPE_BF16) value = ggml_bf16_to_fp32(ggml_fp32_to_bf16(value));
            }
            for (size_t o = 0; o < result.size(); ++o) {
                double expected = 0, bound = 2e-5;
                if (lookup) expected = decoded[size_t(row) * spec.columns + o];
                else for (int64_t i = 0; i < spec.columns; ++i) {
                    const float w = decoded[(size_t(expert) * spec.rows + o) * spec.columns + i];
                    expected += double(w) * values[i];
                    // Bound native Q8 activation rounding/scale error; decode of
                    // the stored weight is exact. No original-F32 quality claim.
                    if (ggml_is_quantized(weight->type)) bound += std::abs(w) * (1.0 / 64 + 0.002 * std::abs(values[i]));
                }
                if (ids && weight->type == GGML_TYPE_F16) bound += 0.003 * (1 + std::abs(expected));
                if (!std::isfinite(result[o]) || std::abs(result[o] - expected) > bound)
                    throw std::runtime_error("Native weight arithmetic differs: " + spec.name + " error=" +
                                             std::to_string(std::abs(result[o] - expected)) + " bound=" + std::to_string(bound));
            }
            ++verified;
        }
        std::cout << verified << " matrix/lookup operations and all uploaded weights verified\n";
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
