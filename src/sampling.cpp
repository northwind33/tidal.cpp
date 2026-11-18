#include "sampling.hpp"
#include "ggml-cpu.h"
#ifdef TIDAL_CUDA
#include "ggml-cuda.h"
#endif
#include <algorithm>
#include <cmath>
#include <numeric>

namespace tidal {
void SamplingConfig::validate() const {
    if ((backend != SamplingBackend::Device && backend != SamplingBackend::CPU) ||
        !std::isfinite(temperature) || temperature < 0 || top_k < -1 ||
        !std::isfinite(top_p) || top_p <= 0 || top_p > 1 ||
        !std::isfinite(min_p) || min_p < 0 || min_p > 1 ||
        !std::isfinite(repeat_penalty) || repeat_penalty <= 0 ||
        !std::isfinite(frequency_penalty) || !std::isfinite(presence_penalty) || penalty_last_n < -1)
        throw std::invalid_argument("Invalid sampling parameters");
}

Sampler::Sampler(int vocab, const SamplingConfig & config, ggml_backend_t backend, int device,
                 Statistics & stats, Profiler & profiler)
    : vocab_(vocab), config_(config), model_device_(!ggml_backend_is_cpu(backend)),
      stats_(stats), profiler_(profiler), backend_name_("CPU"), model_backend_name_(ggml_backend_name(backend)), random_state_(config.seed) {
    config_.validate();
    if (vocab < 1) throw std::invalid_argument("Sampling vocabulary must be positive");
    counts_.resize(vocab);
    if (config.backend == SamplingBackend::Device && model_device_) {
#ifdef TIDAL_CUDA
        if (ggml_backend_is_cuda(backend)) {
            device_ = make_cuda_sampler(vocab, device);
            backend_name_ = ggml_backend_name(backend);
        }
#else
        (void) device;
#endif
        if (!device_) throw std::runtime_error("Device sampling is unsupported on this backend; select the CPU sampler explicitly");
    }
}

void Sampler::reset() {
    std::fill(counts_.begin(), counts_.end(), 0);
    history_.clear(); pending_.clear(); random_state_ = config_.seed;
    if (device_) device_->reset();
}

void Sampler::accept(int32_t token) {
    if (token < 0 || token >= vocab_) throw std::out_of_range("Sampling history token outside vocabulary");
    if (!config_.penalties()) return;
    ++counts_[token]; pending_.insert(token); history_.push_back(token);
    if (config_.penalty_last_n > 0 && history_.size() > size_t(config_.penalty_last_n)) {
        const auto expired = history_.front(); history_.pop_front();
        --counts_[expired]; pending_.insert(expired);
    }
}

void Sampler::rewind(const std::vector<int32_t> & consumed) {
    if (!config_.penalties()) return;
    // Every previously counted token must reach the device counts again.
    for (int32_t token : history_) pending_.insert(token);
    for (int32_t token : history_) counts_[token] = 0;
    history_.clear();
    const size_t keep = config_.penalty_last_n > 0 ? std::min(consumed.size(), size_t(config_.penalty_last_n)) : consumed.size();
    for (size_t i = consumed.size() - keep; i < consumed.size(); ++i) accept(consumed[i]);
}

void Sampler::configure(SamplingConfig config, const std::vector<int32_t> & consumed) {
    config.backend = config_.backend;
    config.validate();
    for (int32_t token : history_) { pending_.insert(token); counts_[token] = 0; }
    history_.clear();
    config_ = config; random_state_ = config.seed;
    rewind(consumed);
}

double Sampler::uniform() {
    // SplitMix64; specified integer operations and 53-bit conversion are identical
    // on CPU/device paths. Only sample(), not prompt ingestion, consumes randomness.
    uint64_t value = (random_state_ += UINT64_C(0x9e3779b97f4a7c15));
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return double((value ^ (value >> 31)) >> 11) * 0x1p-53;
}

int32_t Sampler::sample(ggml_tensor * logits, const std::vector<float> * captured, int position,
                        SamplingDiagnostics * diagnostics) {
    if (logits->type != GGML_TYPE_F32 || ggml_nelements(logits) != vocab_ || !ggml_is_contiguous(logits))
        throw std::invalid_argument("Sampler requires one contiguous F32 vocabulary vector");
    const auto start = std::chrono::steady_clock::now();
    ProfileScope scope(&profiler_, "sampler", "sample", "sampling", position, -1, backend_name_);
    const double draw = config_.temperature > 0 ? uniform() : 0;
    int32_t token;
    if (device_) {
        std::vector<CountUpdate> updates;
        for (int32_t id : pending_) updates.push_back({id, counts_[id]});
        std::unique_ptr<ProfileScope> stage;
        SamplingHook hook;
        if (profiler_.enabled()) hook = [&](const char * name, bool begin) {
            if (begin) {
                const bool transfer = std::string(name) == "H2D" || std::string(name) == "D2H";
                stage = std::make_unique<ProfileScope>(&profiler_, transfer ? "transfer" : "kernel", name,
                                                       "sampling", position, -1, backend_name_);
                if (std::string(name) == "H2D")
                    stage->event.bytes = updates.size() * sizeof(CountUpdate) + (config_.temperature > 0 ? sizeof(double) : 0);
                if (std::string(name) == "D2H") stage->event.bytes = sizeof(SamplingResult);
            } else stage.reset();
        };
        const auto result = device_->run(logits, config_, updates, draw, hook, diagnostics);
        stats_.transfer_ms += device_->transfer_ms;
        pending_.clear();
        const uint64_t up = updates.size() * sizeof(CountUpdate) + (config_.temperature > 0 ? sizeof(double) : 0);
        const uint64_t down = sizeof(SamplingResult) + (diagnostics && config_.temperature > 0 ? diagnostics->ids.size() * (sizeof(int32_t) + sizeof(double)) : 0);
        stats_.host_to_device_bytes += up; stats_.device_to_host_bytes += down;
        stats_.sampling_host_to_device_bytes += up; stats_.sampling_device_to_host_bytes += down;
        if (result.invalid) throw std::runtime_error("Non-finite logits or sampling penalty overflow");
        if (result.token < 0 || result.token >= vocab_) throw std::runtime_error("Invalid device sampling result");
        token = result.token;
    } else {
        std::vector<float> values(vocab_);
        if (captured) {
            if (captured->size() != size_t(vocab_)) throw std::invalid_argument("Wrong captured logit count");
            values = *captured;
        } else {
            ProfileScope transfer(&profiler_, "transfer", model_device_ ? "D2H" : "host_read",
                                  "sampling", position, -1, model_backend_name_);
            transfer.event.bytes = values.size() * sizeof(float);
            const auto tick = std::chrono::steady_clock::now();
            ggml_backend_tensor_get(logits, values.data(), 0, values.size() * sizeof(float));
            stats_.transfer_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tick).count();
            if (model_device_) {
                stats_.device_to_host_bytes += values.size() * sizeof(float);
                stats_.sampling_device_to_host_bytes += values.size() * sizeof(float);
            }
        }
        {
            ProfileScope stage(&profiler_, "cpu", "penalties", "sampling", position, -1, "CPU");
            for (int i = 0; i < vocab_; ++i) {
                float & value = values[i];
                if (!std::isfinite(value)) throw std::runtime_error("Non-finite logits");
                if (counts_[i]) {
                    value = value > 0 ? value / config_.repeat_penalty : value * config_.repeat_penalty;
                    value -= config_.presence_penalty;
                    value -= config_.frequency_penalty * float(counts_[i]);
                }
                if (!std::isfinite(value)) throw std::runtime_error("Sampling penalty overflow");
            }
        }
        token = int32_t(std::max_element(values.begin(), values.end()) - values.begin());
        if (config_.temperature > 0) {
            std::vector<int32_t> ids(vocab_); std::iota(ids.begin(), ids.end(), 0);
            const int count = config_.top_k > 0 ? std::min(config_.top_k, vocab_) : vocab_;
            {
                ProfileScope stage(&profiler_, "cpu", "sort", "sampling", position, -1, "CPU");
                std::partial_sort(ids.begin(), ids.begin() + count, ids.end(), [&](int a, int b) {
                    return values[a] != values[b] ? values[a] > values[b] : a < b;
                });
                ids.resize(count);
            }
            ProfileScope stage(&profiler_, "cpu", "filter_draw", "sampling", position, -1, "CPU");
            std::vector<double> cdf(count);
            double sum = 0;
            for (int i = 0; i < count; ++i) {
                const double mass = std::exp((double(values[ids[i]]) - values[ids[0]]) / config_.temperature);
                if (mass >= config_.min_p) sum += mass;
                cdf[i] = sum;
            }
            const size_t kept = size_t(std::lower_bound(cdf.begin(), cdf.end(), double(config_.top_p) * sum) - cdf.begin()) + 1;
            const double threshold = std::min(draw * cdf[kept - 1], std::nextafter(cdf[kept - 1], 0.0));
            token = ids[size_t(std::upper_bound(cdf.begin(), cdf.begin() + kept, threshold) - cdf.begin())];
            if (diagnostics) *diagnostics = {std::move(ids), std::move(cdf), int(kept)};
        } else if (diagnostics) *diagnostics = {{token}, {1.0}, 1};
        pending_.clear();
    }
    ++stats_.sampling_calls;
    stats_.sampling_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    return token;
}
} // namespace tidal
