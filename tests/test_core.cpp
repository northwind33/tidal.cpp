#include "internal.hpp"
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
