#pragma once
#include "tidal/tidal.hpp"
#include <functional>

struct ggml_tensor;

namespace tidal {
struct CountUpdate { int32_t token, count; };
struct SamplingResult { int32_t token, invalid, kept; };
struct SamplingDiagnostics {
    std::vector<int32_t> ids;
    std::vector<double> cumulative; // Unnormalized masses, before top-p.
    int kept = 0;
};
using SamplingHook = std::function<void(const char *, bool)>;

// Backend implementations own only scratch storage and execution resources.
// Sampling policy, sequence history and random state remain backend independent.
class DeviceSampler {
public:
    virtual ~DeviceSampler() = default;
    virtual void reset() = 0;
    virtual SamplingResult run(const ggml_tensor * logits, const SamplingConfig & config,
        const std::vector<CountUpdate> & updates, double uniform, const SamplingHook & hook,
        SamplingDiagnostics * diagnostics) = 0;
    double transfer_ms = 0;
};
#ifdef TIDAL_CUDA
std::unique_ptr<DeviceSampler> make_cuda_sampler(int vocab, int device);
#endif
} // namespace tidal
