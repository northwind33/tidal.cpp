#include "sampling_device.hpp"
#include "ggml.h"
#include <climits>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace tidal { namespace {
void check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(std::string("CUDA sampler: ") + cudaGetErrorString(result));
}
template<class T> struct Allocation {
    T * ptr = nullptr;
    void allocate(size_t n) { check(cudaMalloc(reinterpret_cast<void **>(&ptr), n * sizeof(T))); }
    ~Allocation() { if (ptr) cudaFree(ptr); }
};
constexpr int threads = 256;
struct Best { float value; int id; };
__device__ Best better(Best a, Best b) {
    return a.value > b.value || (a.value == b.value && a.id < b.id) ? a : b;
}
__global__ void update_counts(int * counts, const CountUpdate * updates, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) counts[updates[i].token] = updates[i].count;
}
__global__ void prepare(const float * logits, const int * counts, float * scores, int * ids,
                        int n, float repeat, float frequency, float presence, SamplingResult * result,
                        Best * blocks) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    Best best{-INFINITY, INT_MAX};
    if (i < n) {
        float value = logits[i];
        bool invalid = !isfinite(value);
        if (counts[i]) {
            value = value > 0 ? value / repeat : value * repeat;
            value -= presence;
            value -= frequency * float(counts[i]);
        }
        invalid |= !isfinite(value);
        if (invalid) { atomicExch(&result->invalid, 1); value = 0; }
        // Canonicalize signed zero so stable radix order agrees with CPU ties.
        if (value == 0) value = 0;
        scores[i] = value; ids[i] = i; best = {value, i};
    }
    __shared__ Best candidates[threads];
    candidates[threadIdx.x] = best;
    __syncthreads();
    for (int stride = threads / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) candidates[threadIdx.x] = better(candidates[threadIdx.x], candidates[threadIdx.x + stride]);
        __syncthreads();
    }
    if (!threadIdx.x) blocks[blockIdx.x] = candidates[0];
}
__global__ void greedy(const Best * blocks, int n, SamplingResult * result) {
    Best best{-INFINITY, INT_MAX};
    for (int i = threadIdx.x; i < n; i += threads) best = better(best, blocks[i]);
    __shared__ Best candidates[threads];
    candidates[threadIdx.x] = best;
    __syncthreads();
    for (int stride = threads / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) candidates[threadIdx.x] = better(candidates[threadIdx.x], candidates[threadIdx.x + stride]);
        __syncthreads();
    }
    if (!threadIdx.x) { result->token = candidates[0].id; result->kept = 1; }
}
__global__ void masses(const float * sorted, double * out, int n, float temperature, float min_p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double mass = exp((double(sorted[i]) - double(sorted[0])) / double(temperature));
        out[i] = mass >= min_p ? mass : 0;
    }
}
// Fixed reduction topology: CUB's decoupled-lookback scan can reassociate
// floating-point sums between runs. Sampling boundaries must not depend on
// block scheduling, so scan local tiles then their totals in a fixed hierarchy.
__global__ void scan_tiles(const double * input, double * output, double * totals, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ double values[threads];
    values[threadIdx.x] = i < n ? input[i] : 0;
    __syncthreads();
    for (int offset = 1; offset < threads; offset *= 2) {
        const double previous = threadIdx.x >= offset ? values[threadIdx.x - offset] : 0;
        __syncthreads();
        values[threadIdx.x] += previous;
        __syncthreads();
    }
    if (i < n) output[i] = values[threadIdx.x];
    if (totals && threadIdx.x == threads - 1) totals[blockIdx.x] = values[threads - 1];
}
__global__ void add_prefix(double * output, const double * totals, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && blockIdx.x) output[i] += totals[blockIdx.x - 1];
}
__global__ void draw(const double * cdf, const int * ids, int n, float top_p,
                     const double * uniform, SamplingResult * result) {
    double cutoff = double(top_p) * cdf[n - 1];
    int lo = 0, hi = n - 1;
    while (lo < hi) { int mid = lo + (hi - lo) / 2; if (cdf[mid] < cutoff) lo = mid + 1; else hi = mid; }
    int kept = lo + 1;
    double threshold = fmin(*uniform * cdf[kept - 1], nextafter(cdf[kept - 1], 0.0));
    lo = 0; hi = kept - 1;
    while (lo < hi) { int mid = lo + (hi - lo) / 2; if (cdf[mid] <= threshold) lo = mid + 1; else hi = mid; }
    result->token = ids[lo]; result->kept = kept;
}
class CUDASampler final : public DeviceSampler {
    int vocab_, device_, blocks_;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t upload_begin_ = nullptr, upload_end_ = nullptr, download_begin_ = nullptr, download_end_ = nullptr;
    Allocation<int> counts_, ids_, sorted_ids_;
    Allocation<float> scores_, sorted_;
    Allocation<double> masses_, cdf_, uniform_, scan_sums_;
    Allocation<Best> blocks;
    Allocation<CountUpdate> updates_;
    Allocation<SamplingResult> result_;
    Allocation<unsigned char> scratch_;
    size_t scratch_bytes_ = 0;
    void stage(const char * name, const SamplingHook & hook, const std::function<void()> & work) {
        if (hook) hook(name, true);
        work(); check(cudaGetLastError());
        if (hook) { check(cudaStreamSynchronize(stream_)); hook(name, false); }
    }
    void scan(const double * input, double * output, int count, double * sums) {
        const int blocks = (count + threads - 1) / threads;
        scan_tiles<<<blocks, threads, 0, stream_>>>(input, output, blocks > 1 ? sums : nullptr, count);
        if (blocks > 1) {
            scan(sums, sums, blocks, sums + blocks);
            add_prefix<<<blocks, threads, 0, stream_>>>(output, sums, count);
        }
    }
    void release_handles() noexcept {
        if (stream_) cudaStreamSynchronize(stream_);
        for (auto event : {upload_begin_, upload_end_, download_begin_, download_end_}) if (event) cudaEventDestroy(event);
        if (stream_) cudaStreamDestroy(stream_);
    }
public:
    CUDASampler(int vocab, int device) : vocab_(vocab), device_(device), blocks_((vocab + threads - 1) / threads) {
        check(cudaSetDevice(device_));
        try {
            check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
            for (auto event : {&upload_begin_, &upload_end_, &download_begin_, &download_end_}) check(cudaEventCreate(event));
            counts_.allocate(vocab); ids_.allocate(vocab); sorted_ids_.allocate(vocab);
            scores_.allocate(vocab); sorted_.allocate(vocab); masses_.allocate(vocab); cdf_.allocate(vocab);
            uniform_.allocate(1); scan_sums_.allocate(2 * blocks_); blocks.allocate(blocks_); updates_.allocate(vocab); result_.allocate(1);
            size_t sort = 0;
            check(cub::DeviceRadixSort::SortPairsDescending(nullptr, sort, scores_.ptr, sorted_.ptr, ids_.ptr, sorted_ids_.ptr, vocab, 0, 32, stream_));
            scratch_bytes_ = sort; scratch_.allocate(scratch_bytes_);
            reset();
        } catch (...) { release_handles(); throw; }
    }
    ~CUDASampler() override { cudaSetDevice(device_); release_handles(); }
    void reset() override {
        check(cudaSetDevice(device_));
        check(cudaMemsetAsync(counts_.ptr, 0, vocab_ * sizeof(int), stream_));
        check(cudaStreamSynchronize(stream_));
    }
    SamplingResult run(const ggml_tensor * logits, const SamplingConfig & config,
        const std::vector<CountUpdate> & updates, double uniform, const SamplingHook & hook,
        SamplingDiagnostics * diagnostics) override {
        check(cudaSetDevice(device_));
        const int count = config.top_k > 0 ? std::min(config.top_k, vocab_) : vocab_;
        stage("H2D", hook, [&] {
            check(cudaEventRecord(upload_begin_, stream_));
            if (!updates.empty()) check(cudaMemcpyAsync(updates_.ptr, updates.data(), updates.size() * sizeof(CountUpdate), cudaMemcpyHostToDevice, stream_));
            if (config.temperature > 0) check(cudaMemcpyAsync(uniform_.ptr, &uniform, sizeof(double), cudaMemcpyHostToDevice, stream_));
            check(cudaEventRecord(upload_end_, stream_));
        });
        stage("penalties", hook, [&] {
            check(cudaMemsetAsync(result_.ptr, 0, sizeof(SamplingResult), stream_));
            if (!updates.empty()) update_counts<<<(updates.size() + threads - 1) / threads, threads, 0, stream_>>>(counts_.ptr, updates_.ptr, int(updates.size()));
            prepare<<<blocks_, threads, 0, stream_>>>(static_cast<const float *>(logits->data), counts_.ptr, scores_.ptr, ids_.ptr, vocab_, config.repeat_penalty,
                                                   config.frequency_penalty, config.presence_penalty, result_.ptr, blocks.ptr);
        });
        if (config.temperature == 0) {
            stage("argmax", hook, [&] { greedy<<<1, threads, 0, stream_>>>(blocks.ptr, blocks_, result_.ptr); });
        } else {
            stage("sort", hook, [&] {
                size_t available = scratch_bytes_;
                check(cub::DeviceRadixSort::SortPairsDescending(scratch_.ptr, available, scores_.ptr, sorted_.ptr, ids_.ptr, sorted_ids_.ptr, vocab_, 0, 32, stream_));
            });
            stage("temperature_min_p", hook, [&] {
                masses<<<(count + threads - 1) / threads, threads, 0, stream_>>>(sorted_.ptr, masses_.ptr, count, config.temperature, config.min_p);
            });
            stage("cumulative_sum", hook, [&] {
                scan(masses_.ptr, cdf_.ptr, count, scan_sums_.ptr);
            });
            stage("top_p_draw", hook, [&] { draw<<<1, 1, 0, stream_>>>(cdf_.ptr, sorted_ids_.ptr, count, config.top_p, uniform_.ptr, result_.ptr); });
        }
        SamplingResult result{};
        stage("D2H", hook, [&] {
            check(cudaEventRecord(download_begin_, stream_));
            check(cudaMemcpyAsync(&result, result_.ptr, sizeof(result), cudaMemcpyDeviceToHost, stream_));
            check(cudaEventRecord(download_end_, stream_));
        });
        check(cudaStreamSynchronize(stream_));
        float up = 0, down = 0;
        check(cudaEventElapsedTime(&up, upload_begin_, upload_end_));
        check(cudaEventElapsedTime(&down, download_begin_, download_end_));
        transfer_ms = up + down;
        if (diagnostics) {
            if (config.temperature == 0) *diagnostics = {{result.token}, {1.0}, 1};
            else {
                diagnostics->ids.resize(count); diagnostics->cumulative.resize(count); diagnostics->kept = result.kept;
                check(cudaMemcpy(diagnostics->ids.data(), sorted_ids_.ptr, count * sizeof(int), cudaMemcpyDeviceToHost));
                check(cudaMemcpy(diagnostics->cumulative.data(), cdf_.ptr, count * sizeof(double), cudaMemcpyDeviceToHost));
            }
        }
        return result;
    }
};
} // namespace
std::unique_ptr<DeviceSampler> make_cuda_sampler(int vocab, int device) { return std::make_unique<CUDASampler>(vocab, device); }
} // namespace tidal
