#pragma once
#include "internal.hpp"
#include "profiler.hpp"
#include "sampling_device.hpp"
#include <deque>
#include <functional>
#include <set>

namespace tidal {
class Sampler {
    int vocab_;
    SamplingConfig config_;
    bool model_device_;
    Statistics & stats_;
    Profiler & profiler_;
    std::string backend_name_, model_backend_name_;
    std::unique_ptr<DeviceSampler> device_;
    std::vector<int32_t> counts_;
    std::deque<int32_t> history_;
    std::set<int32_t> pending_;
    uint64_t random_state_;
    double uniform();
public:
    Sampler(int vocab, const SamplingConfig &, ggml_backend_t backend, int device,
            Statistics &, Profiler &);
    void reset();
    void accept(int32_t token);
    int32_t sample(ggml_tensor * logits, const std::vector<float> * captured, int position,
                   SamplingDiagnostics * diagnostics = nullptr);
    std::string backend_name() const { return backend_name_; }
};
} // namespace tidal
