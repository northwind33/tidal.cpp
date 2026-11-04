#include "internal.hpp"
#include "ggml-cpu.h"
#ifdef TIDAL_CUDA
#include "ggml-cuda.h"
#endif

#include <iomanip>
#include <iostream>

// Test-only driver: each input line names a GGUF matrix followed by its F32
// input vector. The Python oracle supplies inputs and checks native outputs.
// This isolates matvec semantics from recurrent error amplification.
int main(int argc, char ** argv) {
    try {
        if (argc != 3) throw std::invalid_argument("Expected model path and cpu|cuda");
        const std::string requested = argv[2];
        tidal::Backend backend(nullptr, ggml_backend_free);
        if (requested == "cpu") {
            backend.reset(ggml_backend_cpu_init());
            if (backend) ggml_backend_cpu_set_n_threads(backend.get(), 2);
        } else if (requested == "cuda") {
#ifdef TIDAL_CUDA
            backend.reset(ggml_backend_cuda_init(0));
#else
            throw std::runtime_error("CUDA was not compiled");
#endif
        } else {
            throw std::invalid_argument("Expected cpu|cuda");
        }
        if (!backend) throw std::runtime_error("Backend initialization failed");
        tidal::Model model(argv[1], backend.get());
        std::cout << std::setprecision(9);
        for (std::string name; std::cin >> name;) {
            auto * weight = model.at(name);
            std::vector<float> input(size_t(weight->ne[0]));
            for (auto & value : input)
                if (!(std::cin >> value)) throw std::invalid_argument("Truncated input vector");
            tidal::Context ctx(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
            if (!ctx) throw std::runtime_error("Context allocation failed");
            auto * x = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, int64_t(input.size()));
            auto * y = ggml_mul_mat(ctx.get(), weight, x);
            ggml_prec_set_acc(y, GGML_PREC_F32); // Same contract as Engine::mm.
            if (!ggml_backend_supports_op(backend.get(), y))
                throw std::runtime_error("Backend does not support this matvec");
            auto * graph = ggml_new_graph(ctx.get());
            ggml_build_forward_expand(graph, y);
            tidal::Buffer buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()),
                                 ggml_backend_buffer_free);
            if (!buffer) throw std::runtime_error("Buffer allocation failed");
            ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
            if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("Matvec execution failed");
            std::vector<float> output(size_t(ggml_nelements(y)));
            ggml_backend_tensor_get(y, output.data(), 0, output.size() * sizeof(float));
            std::cout << name;
            for (float value : output) std::cout << ' ' << value;
            std::cout << '\n';
        }
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
