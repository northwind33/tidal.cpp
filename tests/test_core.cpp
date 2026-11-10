#include "internal.hpp"
#include "prefill_chunks.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <iostream>

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}
template<class F> void rejects(F && f) {
    try { f(); } catch (const std::exception &) { return; }
    throw std::runtime_error("Expected rejection");
}
int main() {
    try {
        // Changing available memory must reduce only the uncommitted range.
        std::vector<int> committed, attempts, reductions;
        tidal::visit_prefill_chunks(13, 8, true, [&](int first, int n) {
            attempts.push_back(n);
            const int capacity = first < 6 ? 3 : 1;
            if (n > capacity) throw tidal::PrefillAllocationError("simulated allocation pressure");
        }, [&](int first, int n) {
            for (int i=first; i<first+n; ++i) committed.push_back(i);
        }, [&](int, int, int smaller) { reductions.push_back(smaller); });
        require(committed.size()==13, "Adaptive prefill dropped tokens");
        for (int i=0; i<13; ++i) require(committed[i]==i, "Adaptive prefill replayed or reordered tokens");
        require(reductions==std::vector<int>({4,2,1}), "Adaptive prefill did not progressively reduce ranges");
        int calls=0, completed=0;
        auto fail = [&](int, int) { ++calls; throw tidal::PrefillAllocationError("no memory"); };
        auto done = [&](int, int) { ++completed; };
        auto reduced = [](int, int, int) {};
        rejects([&] { tidal::visit_prefill_chunks(8,8,true,fail,done,reduced); });
        require(calls==4 && !completed, "Minimum-size failure did not stop at one token");
        calls=0;
        rejects([&] { tidal::visit_prefill_chunks(8,8,false,fail,done,reduced); });
        require(calls==1, "Explicit prefill size silently shrank");
        calls=0;
        rejects([&] { tidal::visit_prefill_chunks(8,8,true,[&](int,int) { ++calls; },
            [](int,int) { throw tidal::PrefillAllocationError("consumer already changed state"); },reduced); });
        require(calls==1, "Consumer failure replayed committed state");
        calls=0;
        rejects([&] { tidal::visit_prefill_chunks(8,8,true,[&](int,int) {
            ++calls; throw std::runtime_error("execution failure"); },done,reduced); });
        require(calls==1, "Execution failure was treated as retryable allocation pressure");
        require(tidal::expand_blocks({}, 3, 4) == std::vector<int32_t>({0, 1, 2}), "Short tail");
        require(tidal::expand_blocks({2, 0}, 15, 4) ==
            std::vector<int32_t>({0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14}), "Top-k plus incomplete tail");
        rejects([] { tidal::expand_blocks({1, 1}, 8, 4); });
        rejects([] { tidal::expand_blocks({2}, 8, 4); });
        tidal::Config bad;
        bad.q_heads = 3;
        rejects([&] { bad.validate(); });
        // Uniform attention with grouped KV heads and very large equal QK scores:
        // checks stable softmax, GQA head mapping and BF16 value conversion.
        tidal::SparseAttention a(4, 2, 2, 2, 2);
        a.append({1000, 1000, 1000, 1000}, {1, 3, 10, 30});
        a.append({1000, 1000, 1000, 1000}, {3, 5, 30, 50});
        auto v = a.attend(std::vector<float>(8, 1000), {0, 1});
        require(v == std::vector<float>({2, 4, 2, 4, 20, 40, 20, 40}), "Sparse attention values");
        auto selected = a.attend(std::vector<float>(8, 1), {1});
        require(selected[0] == 3 && selected[4] == 30, "Gathered subset");
        rejects([&] { a.attend(std::vector<float>(8), {0, 2}); });
        rejects([&] { a.append(std::vector<float>(4), std::vector<float>(4)); });
        a.reset();
        require(a.size() == 0, "Reset");
        a.append({0, 0, 0, 0}, {4, 8, 12, 16});
        require(a.attend(std::vector<float>(8), {0})[0] == 4, "Reset must exclude stale KV");
        {
            // Grouped, chunked (SIMD when available) attention against a direct
            // double-precision softmax over the same BF16-rounded K/V.
            const int heads = 10, kv = 2, dim = 128, tokens = 300;
            tidal::SparseAttention b(heads, kv, dim, tokens, 4);
            uint32_t seed = 7;
            auto next = [&] { seed = seed * 1664525u + 1013904223u; return float(int(seed >> 9) % 2001 - 1000) / 250.0f; };
            std::vector<std::vector<float>> keys, values;
            for (int t = 0; t < tokens; ++t) {
                std::vector<float> k(size_t(kv) * dim), v(size_t(kv) * dim);
                for (auto & x : k) x = next();
                for (auto & x : v) x = next();
                b.append(k, v); keys.push_back(k); values.push_back(v);
            }
            std::vector<float> query(size_t(heads) * dim);
            for (auto & x : query) x = next() / 8;
            std::vector<int32_t> sparse;
            for (int t = 0; t < tokens; t += 3) sparse.push_back(t);
            for (const auto & selection : {std::vector<int32_t>{5}, sparse}) {
                std::vector<int32_t> all(tokens); std::iota(all.begin(), all.end(), 0);
                for (const auto & chosen : {selection, all}) {
                    const auto out = b.attend(query, chosen);
                    for (int h = 0; h < heads; ++h) {
                        const int g = h / (heads / kv);
                        std::vector<double> score;
                        for (int t : chosen) {
                            double dot = 0;
                            for (int d = 0; d < dim; ++d)
                                dot += double(query[size_t(h) * dim + d]) *
                                    ggml_bf16_to_fp32(ggml_fp32_to_bf16(keys[t][size_t(g) * dim + d]));
                            score.push_back(dot / std::sqrt(double(dim)));
                        }
                        const double top = *std::max_element(score.begin(), score.end());
                        double sum = 0; for (auto & x : score) sum += (x = std::exp(x - top));
                        for (int d = 0; d < dim; ++d) {
                            double expected = 0;
                            for (size_t i = 0; i < chosen.size(); ++i)
                                expected += score[i] / sum * ggml_bf16_to_fp32(ggml_fp32_to_bf16(values[chosen[i]][size_t(g) * dim + d]));
                            require(std::abs(out[size_t(h) * dim + d] - expected) <= 1e-4 + 1e-4 * std::abs(expected),
                                    "Grouped chunked attention differs from the reference softmax");
                        }
                    }
                }
            }
        }
        {
            // Every task of every run executes exactly once, including runs with
            // fewer tasks than workers and back-to-back runs on the same pool.
            tidal::WorkerPool pool(4);
            std::vector<std::atomic<int>> hits(64);
            for (int round = 0; round < 2000; ++round) {
                const int tasks = 1 + round % 64;
                for (int i = 0; i < tasks; ++i) hits[i].store(0);
                pool.run(tasks, [&](int i) { hits[i].fetch_add(1); });
                for (int i = 0; i < tasks; ++i) require(hits[i].load() == 1, "Worker pool lost or repeated a task");
            }
        }
        std::cout << "core tests passed\n";
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
