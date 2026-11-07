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
        // Compare exact replay with the same partition. Batched CUDA reductions
        // need not be bit-identical to single-token kernels; independent numeric
        // and cross-partition checks live in the prefill suite.
        std::vector<tidal::Step> expected;
        for (size_t i=0;i<tokens.size();) {
            if (i%3==0) { expected.push_back(engine.decode(tokens[i++])); }
            else {
                const size_t end=std::min(tokens.size(),i+4);
                auto chunk=engine.prefill({tokens.begin()+i,tokens.begin()+end});
                expected.insert(expected.end(),chunk.begin(),chunk.end());i=end;
            }
        }
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
        // Compact decode retains the GPU logits until sample(), and reset must
        // discard both consumed-token history and random state.
        tidal::Options options{argv[2], 0, 2};
        options.sampling.temperature = .8f;
        options.sampling.top_k = 31;
        options.sampling.top_p = .9f;
        options.sampling.repeat_penalty = 1.2f;
        options.sampling.penalty_last_n = 4;
        options.sampling.seed = 71;
        tidal::Engine sampled(argv[1], options);
        bool rejected = false;
        try { sampled.sample(); } catch (const std::logic_error &) { rejected = true; }
        if (!rejected) throw std::runtime_error("Sampling before decode succeeded");
        std::vector<int32_t> draws;
        for (int pass = 0; pass < 2; ++pass) {
            sampled.reset();
            sampled.decode(1, !pass); sampled.prefill({2,3,2}, !pass);
            for (int i = 0; i < 5; ++i) {
                const auto token = sampled.sample();
                if (!pass) draws.push_back(token);
                else if (draws[i] != token) throw std::runtime_error("Capture/chunk/reset changed sampling");
                sampled.decode(token, !pass);
            }
        }
        std::cout << "Chunked prefill/decode, rejected input and partial reset passed\n";
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
