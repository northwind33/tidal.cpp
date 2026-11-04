#include "tidal/tidal.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace {
void equal(const tidal::Step & a, const tidal::Step & b) {
    if (a.position != b.position || a.logits != b.logits || a.selections.size() != b.selections.size())
        throw std::runtime_error("Sequence replay differs");
    for (size_t i = 0; i < a.selections.size(); ++i)
        if (a.selections[i].layer != b.selections[i].layer || a.selections[i].tokens != b.selections[i].tokens)
            throw std::runtime_error("QSA replay differs");
}
template<class F> void rejects(F f) {
    try { f(); } catch (const std::out_of_range &) { return; }
    throw std::runtime_error("Invalid sequence input accepted");
}
}

int main(int argc, char ** argv) {
    try {
        if (argc != 3) throw std::runtime_error("Usage: tidal-state-test model.gguf cpu|cuda");
        tidal::Engine engine(argv[1], {argv[2], 0, 2});
        const std::vector<int32_t> tokens{0, 2, 7, 7, 11, 3, 255, 4, 1, 2, 7, 0, 0, 11, 5, 3, 6, 4, 9, 10};
        const auto expected = engine.prefill(tokens);
        for (int consumed : {0, 3, 13}) {
            engine.reset();
            for (int i = 0; i < consumed; ++i) engine.decode(42);
            engine.reset(); // Discard a partial sequence, including both PLE caches.
            rejects([&] { engine.decode(-1); });
            rejects([&] { engine.prefill({1, engine.config().vocab}); });
            rejects([&] { engine.prefill(std::vector<int32_t>(engine.config().context + 1, 1)); });
            for (size_t i = 0; i < tokens.size();) {
                if (i % 3 == 0) {
                    equal(expected[i], engine.decode(tokens[i]));
                    ++i;
                } else {
                    const size_t end = std::min(tokens.size(), i + 4);
                    auto chunk = engine.prefill(std::vector<int32_t>(tokens.begin() + i, tokens.begin() + end));
                    for (const auto & step : chunk) equal(expected[i++], step);
                }
            }
        }
        std::cout << "Chunked prefill/decode, rejected input and partial reset passed\n";
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
