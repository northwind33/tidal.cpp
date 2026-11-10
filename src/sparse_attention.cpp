#include "internal.hpp"
#include "profiler.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <thread>
#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace tidal {
namespace {
inline void _mm_pause_or_yield() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#else
    std::this_thread::yield();
#endif
}
} // namespace
namespace {
// Grouped-query kernels: the R query heads sharing one KV head are processed
// together, so each K/V row is read once per layer instead of once per query head.
// KV elements are 16-bit BF16 or FP16 bit patterns.
using Dot = void (*)(const float * q, int rows, int dim, const uint16_t * k, float * out);
using Axpy = void (*)(float * acc, const float * p, int rows, int dim, const uint16_t * v);

template<bool F16> inline float widen(uint16_t x) {
    if constexpr (F16) return ggml_fp16_to_fp32(ggml_fp16_t(x));
    else { ggml_bf16_t b; b.bits = x; return ggml_bf16_to_fp32(b); }
}
template<bool F16> void dot_scalar(const float * q, int rows, int dim, const uint16_t * k, float * out) {
    for (int r = 0; r < rows; ++r) {
        float dot = 0;
        for (int d = 0; d < dim; ++d) dot += q[size_t(r) * dim + d] * widen<F16>(k[d]);
        out[r] = dot;
    }
}
template<bool F16> void axpy_scalar(float * acc, const float * p, int rows, int dim, const uint16_t * v) {
    for (int d = 0; d < dim; ++d) {
        const float value = widen<F16>(v[d]);
        for (int r = 0; r < rows; ++r) acc[size_t(r) * dim + d] += p[r] * value;
    }
}

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define TIDAL_X86_KERNELS 1
// FP16 uses the F16C conversion; BF16 widens by a 16-bit shift.
template<bool F16> __attribute__((target("avx2,fma,f16c"))) inline __m256 load16(const uint16_t * x) {
    const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i *>(x));
    if constexpr (F16) return _mm256_cvtph_ps(raw);
    else return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(raw), 16));
}
__attribute__((target("avx2,fma"))) inline float hsum(__m256 x) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}
template<bool F16> __attribute__((target("avx2,fma,f16c"))) void dot_avx2(const float * q, int rows, int dim, const uint16_t * k, float * out) {
    for (int r0 = 0; r0 < rows; r0 += 8) {
        const int n = std::min(8, rows - r0);
        __m256 acc[8];
        for (int r = 0; r < n; ++r) acc[r] = _mm256_setzero_ps();
        for (int d = 0; d < dim; d += 8) {
            const __m256 key = load16<F16>(k + d);
            for (int r = 0; r < n; ++r)
                acc[r] = _mm256_fmadd_ps(_mm256_loadu_ps(q + size_t(r0 + r) * dim + d), key, acc[r]);
        }
        for (int r = 0; r < n; ++r) out[r0 + r] = hsum(acc[r]);
    }
}
template<bool F16> __attribute__((target("avx2,fma,f16c"))) void axpy_avx2(float * acc, const float * p, int rows, int dim, const uint16_t * v) {
    for (int d = 0; d < dim; d += 8) {
        const __m256 value = load16<F16>(v + d);
        for (int r = 0; r < rows; ++r) {
            float * out = acc + size_t(r) * dim + d;
            _mm256_storeu_ps(out, _mm256_fmadd_ps(_mm256_set1_ps(p[r]), value, _mm256_loadu_ps(out)));
        }
    }
}
#endif

struct Kernels { Dot dot; Axpy axpy; const char * name; };
Kernels kernels(int dim, bool f16) {
#ifdef TIDAL_X86_KERNELS
    static const bool avx2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
    if (avx2 && dim % 8 == 0)
        return f16 ? Kernels{dot_avx2<true>, axpy_avx2<true>, "avx2"} : Kernels{dot_avx2<false>, axpy_avx2<false>, "avx2"};
#endif
    return f16 ? Kernels{dot_scalar<true>, axpy_scalar<true>, "scalar"} : Kernels{dot_scalar<false>, axpy_scalar<false>, "scalar"};
}
constexpr float fp16_max = 65504.0f;
} // namespace

WorkerPool::WorkerPool(int threads) {
    if (threads <= 0) throw std::invalid_argument("Worker pool needs at least one thread");
    for (int i = 1; i < threads; ++i) workers_.emplace_back([this] { work(); });
}

WorkerPool::~WorkerPool() {
    { std::lock_guard lock(mutex_); stop_ = true; }
    wake_.notify_all();
    for (auto & worker : workers_) worker.join();
}

void WorkerPool::drain(uint64_t generation, int tasks, const std::function<void(int)> * job) {
    uint64_t ticket = ticket_.load(std::memory_order_acquire);
    while ((ticket >> 32) == generation && int(ticket & 0xffffffffu) < tasks) {
        if (!ticket_.compare_exchange_weak(ticket, ticket + 1, std::memory_order_acq_rel, std::memory_order_acquire)) continue;
        (*job)(int(ticket & 0xffffffffu));
        finished_.fetch_add(1, std::memory_order_acq_rel);
        ticket = ticket_.load(std::memory_order_acquire);
    }
}

void WorkerPool::work() {
    uint64_t seen = 0;
    for (;;) {
        uint64_t generation; int tasks; const std::function<void(int)> * job;
        {
            std::unique_lock lock(mutex_);
            wake_.wait(lock, [&] { return stop_ || generation_ != seen; });
            if (stop_) return;
            seen = generation = generation_; tasks = tasks_; job = job_;
        }
        drain(generation, tasks, job);
    }
}

void WorkerPool::run(int tasks, const std::function<void(int)> & fn) {
    if (tasks <= 0) return;
    if (workers_.empty() || tasks == 1) { for (int i = 0; i < tasks; ++i) fn(i); return; }
    uint64_t generation;
    {
        std::lock_guard lock(mutex_);
        // Generations live in the ticket's upper 32 bits; 0 is the workers' initial view.
        generation = (generation_ + 1) & 0xffffffffu;
        if (!generation) generation = 1;
        generation_ = generation;
        job_ = &fn; tasks_ = tasks;
        finished_.store(0, std::memory_order_relaxed);
        ticket_.store(generation << 32, std::memory_order_release);
    }
    // Wake only as many workers as there are tasks beyond the caller's share.
    const int helpers = std::min<int>(tasks - 1, int(workers_.size()));
    if (helpers == int(workers_.size())) wake_.notify_all();
    else for (int i = 0; i < helpers; ++i) wake_.notify_one();
    drain(generation, tasks, &fn);
    // A claimed task always finishes before run() returns, so job pointers held
    // by workers are dereferenced only while fn is alive.
    while (finished_.load(std::memory_order_acquire) != tasks) _mm_pause_or_yield();
}

SparseAttention::SparseAttention(int heads, int kv_heads, int dim, int capacity, int threads,
                                 std::shared_ptr<WorkerPool> pool, ggml_type type)
    : heads_(heads), kv_heads_(kv_heads), dim_(dim), capacity_(capacity), threads_(threads), type_(type), pool_(std::move(pool)) {
    if (heads <= 0 || kv_heads <= 0 || heads % kv_heads || dim <= 0 || capacity <= 0 || threads <= 0)
        throw std::invalid_argument("Invalid sparse-attention dimensions");
    if (type != GGML_TYPE_BF16 && type != GGML_TYPE_F16) throw std::invalid_argument("CPU KV must be BF16 or F16");
    if (!pool_) pool_ = std::make_shared<WorkerPool>(threads);
    keys_.resize(size_t(capacity) * kv_heads * dim);
    values_.resize(keys_.size());
}

void SparseAttention::append(const std::vector<float> & key, const std::vector<float> & value) {
    const auto width = size_t(kv_heads_) * dim_;
    if (key.size() != width || value.size() != width) throw std::invalid_argument("Wrong KV width");
    if (size_ >= capacity_) throw std::out_of_range("CPU KV cache is full");
    auto * k = keys_.data() + size_t(size_) * width, * v = values_.data() + size_t(size_) * width;
    if (type_ == GGML_TYPE_F16) {
        // Reject before storing: an infinite FP16 K/V would silently corrupt attention.
        for (size_t i = 0; i < width; ++i)
            if (!(std::abs(key[i]) <= fp16_max) || !(std::abs(value[i]) <= fp16_max))
                throw std::overflow_error("KV value " + std::to_string(std::abs(key[i]) <= fp16_max ? value[i] : key[i]) +
                                          " is outside the FP16 range");
        ggml_fp32_to_fp16_row(key.data(), reinterpret_cast<ggml_fp16_t *>(k), int64_t(width));
        ggml_fp32_to_fp16_row(value.data(), reinterpret_cast<ggml_fp16_t *>(v), int64_t(width));
    } else {
        ggml_fp32_to_bf16_row_ref(key.data(), reinterpret_cast<ggml_bf16_t *>(k), int64_t(width));
        ggml_fp32_to_bf16_row_ref(value.data(), reinterpret_cast<ggml_bf16_t *>(v), int64_t(width));
    }
    ++size_;
}

void SparseAttention::append_raw(const uint16_t * key, const uint16_t * value, int tokens) {
    if (tokens < 0 || tokens > capacity_ - size_) throw std::out_of_range("CPU KV cache is full");
    const size_t width = size_t(kv_heads_) * dim_, count = size_t(tokens) * width;
    if (type_ == GGML_TYPE_F16)
        // An all-ones FP16 exponent is Inf or NaN: the device conversion overflowed.
        for (size_t i = 0; i < count; ++i)
            if ((key[i] & 0x7c00) == 0x7c00 || (value[i] & 0x7c00) == 0x7c00)
                throw std::overflow_error("KV row " + std::to_string(size_ + int(i / width)) + " is outside the FP16 range");
    std::memcpy(keys_.data() + size_t(size_) * width, key, count * sizeof(uint16_t));
    std::memcpy(values_.data() + size_t(size_) * width, value, count * sizeof(uint16_t));
    size_ += tokens;
}

std::vector<float> SparseAttention::attend(const std::vector<float> & query,
                                         const std::vector<int32_t> & selected, Profiler * profiler,
                                         int position, int layer, const std::string & phase) const {
    if (query.size() != size_t(heads_) * dim_ || selected.empty())
        throw std::invalid_argument("Wrong query width or empty attention selection");
    int previous = -1;
    for (auto token : selected) {
        if (token <= previous || token >= size_) throw std::invalid_argument("Non-causal, duplicate or unsorted KV index");
        previous = token;
    }
    const int group = heads_ / kv_heads_; // grouped, not tiled, query heads
    const int64_t n = int64_t(selected.size());
    const auto kernel = kernels(dim_, type_ == GGML_TYPE_F16);
    const int workers = std::min(threads_, pool_->threads());
    // Split each KV head's tokens so all workers stream KV; chunks are merged
    // with the usual max-rescaled softmax combination.
    const int chunks = int(std::max<int64_t>(1, std::min<int64_t>((n + 31) / 32, (2 * workers + kv_heads_ - 1) / kv_heads_)));
    const int tasks = kv_heads_ * chunks;
    const size_t stride = size_t(group) * (dim_ + 2);
    std::vector<float> partial(size_t(tasks) * stride);
    const float scale = std::sqrt(float(dim_));
    auto task = [&](int t) {
        const int g = t / chunks, c = t % chunks;
        const int64_t first = n * c / chunks, last = n * (c + 1) / chunks, count = last - first;
        const float * q = query.data() + size_t(g) * group * dim_;
        thread_local std::vector<float> scores;
        scores.resize(size_t(count) * group);
        ProfileScope qk(profiler, "cpu_head", "qk", phase, position, layer, "CPU", g * group);
        qk.event.bytes = size_t(count) * dim_ * sizeof(uint16_t);
        for (int64_t i = 0; i < count; ++i) {
            float * out = scores.data() + size_t(i) * group;
            kernel.dot(q, group, dim_, keys_.data() + (size_t(selected[first + i]) * kv_heads_ + g) * dim_, out);
            for (int r = 0; r < group; ++r) out[r] /= scale;
        }
        qk.finish();
        ProfileScope softmax(profiler, "cpu_head", "softmax", phase, position, layer, "CPU", g * group);
        float * result = partial.data() + size_t(t) * stride;
        float * maximum = result, * sum = result + group, * acc = result + 2 * group;
        for (int r = 0; r < group; ++r) { maximum[r] = -std::numeric_limits<float>::infinity(); sum[r] = 0; }
        for (int64_t i = 0; i < count; ++i)
            for (int r = 0; r < group; ++r) maximum[r] = std::max(maximum[r], scores[size_t(i) * group + r]);
        for (int64_t i = 0; i < count; ++i)
            for (int r = 0; r < group; ++r) {
                float & p = scores[size_t(i) * group + r];
                p = std::exp(p - maximum[r]); sum[r] += p;
            }
        softmax.finish();
        ProfileScope weighted(profiler, "cpu_head", "weighted_v", phase, position, layer, "CPU", g * group);
        weighted.event.bytes = size_t(count) * dim_ * sizeof(uint16_t);
        std::fill(acc, acc + size_t(group) * dim_, 0.0f);
        for (int64_t i = 0; i < count; ++i)
            kernel.axpy(acc, scores.data() + size_t(i) * group, group, dim_,
                        values_.data() + (size_t(selected[first + i]) * kv_heads_ + g) * dim_);
    };
    {
        ProfileScope dispatch(profiler, "host", "worker_dispatch", phase, position, layer, "CPU");
        pool_->run(tasks, task);
    }
    std::vector<float> result(query.size());
    for (int g = 0; g < kv_heads_; ++g)
        for (int r = 0; r < group; ++r) {
            float maximum = -std::numeric_limits<float>::infinity(), denominator = 0;
            for (int c = 0; c < chunks; ++c) maximum = std::max(maximum, partial[size_t(g * chunks + c) * stride + r]);
            float * out = result.data() + size_t(g * group + r) * dim_;
            for (int c = 0; c < chunks; ++c) {
                const float * part = partial.data() + size_t(g * chunks + c) * stride;
                const float weight = std::exp(part[r] - maximum);
                denominator += weight * part[group + r];
                const float * acc = part + 2 * group + size_t(r) * dim_;
                for (int d = 0; d < dim_; ++d) out[d] += weight * acc[d];
            }
            for (int d = 0; d < dim_; ++d) out[d] /= denominator;
        }
    return result;
}

std::vector<int32_t> expand_blocks(const std::vector<int32_t> & blocks, int visible, int block_size) {
    if (visible <= 0 || block_size <= 0) throw std::invalid_argument("Invalid QSA block geometry");
    const int complete = visible / block_size;
    std::vector<int32_t> selected;
    for (int b : blocks) {
        if (b < 0 || b >= complete) throw std::out_of_range("Invalid QSA block ID");
        for (int t = 0; t < block_size; ++t) selected.push_back(b * block_size + t);
    }
    for (int t = complete * block_size; t < visible; ++t) selected.push_back(t);
    std::sort(selected.begin(), selected.end());
    if (std::adjacent_find(selected.begin(), selected.end()) != selected.end())
        throw std::invalid_argument("Duplicate QSA block ID");
    if (selected.empty()) throw std::invalid_argument("Empty QSA selection");
    return selected;
}
} // namespace tidal
