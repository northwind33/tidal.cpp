// Fixed-logit probe for the independent NumPy sampler oracle.
#include "sampling.hpp"
#include "ggml-cpu.h"
#ifdef TIDAL_CUDA
#include "ggml-cuda.h"
#endif
#include "sampling_options.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

template<class T> void array(const std::vector<T> & values) {
    std::cout << '[';
    for (size_t i = 0; i < values.size(); ++i) { if (i) std::cout << ','; std::cout << values[i]; }
    std::cout << ']';
}
int main(int argc, char ** argv) {
    try {
        Args args(argc, argv, sampling_options({"--input", "--backend", "--history", "--draws", "--feedback", "--diagnostics"}));
        const auto config = sampling_config(args);
        std::ifstream in(args.get("--input"), std::ios::binary | std::ios::ate);
        if (!in || in.tellg() <= 0 || in.tellg() % sizeof(float)) throw std::runtime_error("Invalid logits file");
        std::vector<float> logits(size_t(in.tellg()) / sizeof(float));
        in.seekg(0); in.read(reinterpret_cast<char *>(logits.data()), logits.size() * sizeof(float));
        tidal::Backend backend(nullptr, ggml_backend_free);
        if (args.get("--backend", "cpu") == "cpu") backend.reset(ggml_backend_cpu_init());
#ifdef TIDAL_CUDA
        else backend.reset(ggml_backend_cuda_init(0));
#endif
        if (!backend) throw std::runtime_error("Backend unavailable");
        tidal::Context ctx(ggml_init({65536, nullptr, true}), ggml_free);
        if (!ctx) throw std::runtime_error("Context allocation failed");
        auto * tensor = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, logits.size());
        tidal::Buffer buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()), ggml_backend_buffer_free);
        if (!buffer) throw std::runtime_error("Allocation failed");
        ggml_backend_tensor_set(tensor, logits.data(), 0, logits.size() * sizeof(float));
        tidal::Statistics stats;
        tidal::Profiler profiler(tidal::ProfileMode::Off, false, 1000);
        tidal::Sampler sampler(int(logits.size()), config, backend.get(), 0, stats, profiler);
        const bool diagnostics = args.integer("--diagnostics", 1) != 0;
        std::cout << std::setprecision(17) << '[';
        for (int replay = 0; replay < 2; ++replay) {
            if (replay) std::cout << ',';
            sampler.reset();
            std::istringstream history(args.get("--history"));
            for (std::string token; std::getline(history, token, ',');) sampler.accept(number(token));
            std::cout << '[';
            for (int i = 0; i < args.integer("--draws", 1); ++i) {
                tidal::SamplingDiagnostics d;
                const auto token = sampler.sample(tensor, nullptr, i, diagnostics ? &d : nullptr);
                if (i) std::cout << ',';
                std::cout << "{\"token\":" << token;
                if (diagnostics) {
                    std::cout << ",\"ids\":"; array(d.ids);
                    std::cout << ",\"cdf\":"; array(d.cumulative);
                    std::cout << ",\"kept\":" << d.kept;
                }
                std::cout << '}';
                if (args.integer("--feedback", 0)) sampler.accept(token);
            }
            std::cout << ']';
        }
        std::cout << "]\n";
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
