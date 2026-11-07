#include "internal.hpp"
#include "prefill_chunks.hpp"
#include <cmath>
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
        std::cout << "core tests passed\n";
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
