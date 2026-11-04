#include "internal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <thread>

namespace tidal {
SparseAttention::SparseAttention(int heads, int kv_heads, int dim, int capacity, int threads)
    : heads_(heads), kv_heads_(kv_heads), dim_(dim), capacity_(capacity), threads_(threads) {
    if (heads <= 0 || kv_heads <= 0 || heads % kv_heads || dim <= 0 || capacity <= 0 || threads <= 0)
        throw std::invalid_argument("Invalid sparse-attention dimensions");
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

std::vector<float> SparseAttention::attend(const std::vector<float> & query,
                                         const std::vector<int32_t> & selected) const {
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
            float denominator = 0;
            for (float & score : scores) { score = std::exp(score - maximum); denominator += score; }
            for (size_t i = 0; i < selected.size(); ++i) {
                const size_t offset = (size_t(selected[i]) * kv_heads_ + kv) * dim_;
                const float probability = scores[i] / denominator;
                for (int d = 0; d < dim_; ++d)
                    result[size_t(h) * dim_ + d] += probability * ggml_bf16_to_fp32(values_[offset + d]);
            }
        }
    };
    const int workers = std::min(threads_, heads_);
    std::vector<std::jthread> pool;
    for (int w = 1; w < workers; ++w) pool.emplace_back(worker, w, workers);
    worker(0, workers);
    // Join before moving result into the return value.
    for (auto & thread : pool) thread.join();
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
