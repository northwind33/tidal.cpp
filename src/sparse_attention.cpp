#include "internal.hpp"
#include "profiler.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <thread>

namespace tidal {
WorkerPool::WorkerPool(int threads) {
    if (threads <= 0) throw std::invalid_argument("Worker pool needs at least one thread");
    for (int i = 1; i < threads; ++i) workers_.emplace_back([this] { work(); });
}

WorkerPool::~WorkerPool() {
    { std::lock_guard lock(mutex_); stop_ = true; }
    wake_.notify_all();
    for (auto & worker : workers_) worker.join();
}

void WorkerPool::drain() {
    for (int task; (task = next_.fetch_add(1, std::memory_order_relaxed)) < tasks_; ) {
        (*job_)(task);
        finished_.fetch_add(1, std::memory_order_release);
    }
}

void WorkerPool::work() {
    uint64_t seen = 0;
    for (;;) {
        {
            std::unique_lock lock(mutex_);
            wake_.wait(lock, [&] { return stop_ || generation_ != seen; });
            if (stop_) return;
            seen = generation_;
            ++acknowledged_; ++busy_;
        }
        drain();
        { std::lock_guard lock(mutex_); --busy_; }
        done_.notify_one();
    }
}

void WorkerPool::run(int tasks, const std::function<void(int)> & fn) {
    if (tasks <= 0) return;
    if (workers_.empty() || tasks == 1) { for (int i = 0; i < tasks; ++i) fn(i); return; }
    {
        std::lock_guard lock(mutex_);
        job_ = &fn; tasks_ = tasks;
        next_.store(0, std::memory_order_relaxed); finished_.store(0, std::memory_order_relaxed);
        acknowledged_ = 0;
        ++generation_;
    }
    wake_.notify_all();
    drain();
    // Return only after every worker has observed this generation and left
    // drain(), so no worker can carry a stale view of job_ into the next run.
    std::unique_lock lock(mutex_);
    done_.wait(lock, [&] {
        return finished_.load(std::memory_order_acquire) == tasks_ && busy_ == 0 && acknowledged_ == int(workers_.size());
    });
    job_ = nullptr;
}

SparseAttention::SparseAttention(int heads, int kv_heads, int dim, int capacity, int threads,
                                 std::shared_ptr<WorkerPool> pool)
    : heads_(heads), kv_heads_(kv_heads), dim_(dim), capacity_(capacity), threads_(threads), pool_(std::move(pool)) {
    if (heads <= 0 || kv_heads <= 0 || heads % kv_heads || dim <= 0 || capacity <= 0 || threads <= 0)
        throw std::invalid_argument("Invalid sparse-attention dimensions");
    if (!pool_) pool_ = std::make_shared<WorkerPool>(threads);
    keys_.resize(size_t(capacity) * kv_heads * dim);
    values_.resize(keys_.size());
}

void SparseAttention::append(const std::vector<float> & key, const std::vector<float> & value) {
    const auto width = size_t(kv_heads_) * dim_;
    if (key.size() != width || value.size() != width) throw std::invalid_argument("Wrong KV width");
    if (size_ >= capacity_) throw std::out_of_range("CPU KV cache is full");
    ggml_fp32_to_bf16_row_ref(key.data(), keys_.data() + size_t(size_) * width, int64_t(width));
    ggml_fp32_to_bf16_row_ref(value.data(), values_.data() + size_t(size_) * width, int64_t(width));
    ++size_;
}

void SparseAttention::append_bf16(const ggml_bf16_t * key, const ggml_bf16_t * value, int tokens) {
    if (tokens < 0 || tokens > capacity_ - size_) throw std::out_of_range("CPU KV cache is full");
    const size_t width = size_t(kv_heads_) * dim_;
    std::memcpy(keys_.data() + size_t(size_) * width, key, size_t(tokens) * width * sizeof(ggml_bf16_t));
    std::memcpy(values_.data() + size_t(size_) * width, value, size_t(tokens) * width * sizeof(ggml_bf16_t));
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
    std::vector<float> result(query.size());
    auto worker = [&](int first, int stride) {
        std::vector<float> scores(selected.size());
        for (int h = first; h < heads_; h += stride) {
            ProfileScope qk(profiler, "cpu_head", "qk", phase, position, layer, "CPU", h);
            qk.event.bytes = selected.size() * dim_ * sizeof(ggml_bf16_t);
            const int kv = h / (heads_ / kv_heads_); // grouped, not tiled, query heads
            float maximum = -std::numeric_limits<float>::infinity();
            for (size_t i = 0; i < selected.size(); ++i) {
                const size_t offset = (size_t(selected[i]) * kv_heads_ + kv) * dim_;
                float dot = 0;
                for (int d = 0; d < dim_; ++d)
                    dot += query[size_t(h) * dim_ + d] * ggml_bf16_to_fp32(keys_[offset + d]);
                scores[i] = dot / std::sqrt(float(dim_));
                maximum = std::max(maximum, scores[i]);
            }
            qk.finish();
            ProfileScope softmax(profiler, "cpu_head", "softmax", phase, position, layer, "CPU", h);
            float denominator = 0;
            for (float & score : scores) { score = std::exp(score - maximum); denominator += score; }
            softmax.finish();
            ProfileScope weighted(profiler, "cpu_head", "weighted_v", phase, position, layer, "CPU", h);
            weighted.event.bytes = selected.size() * dim_ * sizeof(ggml_bf16_t);
            for (size_t i = 0; i < selected.size(); ++i) {
                const size_t offset = (size_t(selected[i]) * kv_heads_ + kv) * dim_;
                const float probability = scores[i] / denominator;
                for (int d = 0; d < dim_; ++d)
                    result[size_t(h) * dim_ + d] += probability * ggml_bf16_to_fp32(values_[offset + d]);
            }
        }
    };
    const int workers = std::min({threads_, heads_, pool_->threads()});
    ProfileScope dispatch(profiler, "host", "worker_dispatch", phase, position, layer, "CPU");
    pool_->run(workers, [&](int w) { worker(w, workers); });
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
