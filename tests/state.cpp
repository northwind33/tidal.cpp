#include "tidal/tidal.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void equal(const tidal::Step & a, const tidal::Step & b) {
    if (a.position != b.position || a.logits != b.logits || a.selections.size() != b.selections.size())
        throw std::runtime_error("Sequence replay differs");
    for (size_t i = 0; i < a.selections.size(); ++i)
        if (a.selections[i].layer != b.selections[i].layer || a.selections[i].tokens != b.selections[i].tokens)
            throw std::runtime_error("QSA replay differs");
}
void require(bool value, const char * error) { if (!value) throw std::runtime_error(error); }
void equal_steps(const std::vector<tidal::Step> & a, const std::vector<tidal::Step> & b) {
    require(a.size() == b.size(), "Prompt cache evaluated a different number of positions");
    for (size_t i = 0; i < a.size(); ++i) equal(a[i], b[i]);
}
std::vector<int32_t> join(std::vector<int32_t> a, const std::vector<int32_t> & b) {
    a.insert(a.end(), b.begin(), b.end()); return a;
}
// Reuse and rollback must reproduce, bit for bit, a fresh engine that runs the
// same prefill/decode partition: the restored state is the state at that position.
void prompt_cache(const char * model, tidal::Options options) {
    options.prefill_chunk_tokens = 8; options.prefill_attention_tile = 5; // Independent of free memory.
    options.sampling.repeat_penalty = 1.3f; options.sampling.penalty_last_n = 8; // Rewinds penalty history.
    tidal::Engine cached(model, options), fresh(model, options);
    const auto & c = cached.config();
    const int length = std::min(37, c.context - 11); // Not a block multiple: restores a partial QSA ring.
    std::vector<int32_t> prompt, extra, other;
    for (int i = 0; i < length; ++i) prompt.push_back((i * 7 + 3) % c.vocab);
    for (int i = 0; i < 6; ++i) extra.push_back((i * 5 + 1) % c.vocab);
    require(cached.prefill_cached(prompt).size() == prompt.size() && !cached.statistics().prompt_cache_reused_tokens,
            "Cold prompt cache reused state");
    std::vector<int32_t> generated;
    for (int i = 0; i < 5; ++i) { generated.push_back(cached.sample()); cached.decode(generated.back()); }
    // Extension of the consumed sequence (prompt + generated + new turn).
    const auto extended = cached.prefill_cached(join(join(prompt, generated), extra));
    require(cached.statistics().prompt_cache_reused_tokens == prompt.size() + generated.size() &&
            !cached.statistics().prompt_cache_restores, "Extension did not reuse the consumed sequence");
    fresh.prefill(prompt); for (auto t : generated) fresh.decode(t);
    equal_steps(fresh.prefill(extra), extended);
    require(cached.tokens() == join(join(prompt, generated), extra), "Consumed token sequence");
    // Divergence after the prompt: roll back to the prompt-end snapshot.
    for (int i = 0; i < 6; ++i) other.push_back((generated[0] + 1 + i) % c.vocab);
    const auto diverged = cached.prefill_cached(join(prompt, other));
    require(cached.statistics().prompt_cache_restores == 1 &&
            cached.statistics().prompt_cache_reused_tokens == 2 * prompt.size() + generated.size(), "Rollback reuse");
    fresh.reset(); fresh.prefill(prompt);
    equal_steps(fresh.prefill(other), diverged);
    require(cached.sample() == fresh.sample(), "Rollback changed sampling");
    // Regeneration of a prompt that ends at a snapshot evaluates nothing.
    cached.decode(3); cached.decode(4);
    require(cached.prefill_cached(prompt).empty() && cached.tokens() == prompt, "Snapshot prompt was re-evaluated");
    fresh.reset(); fresh.prefill(prompt);
    require(cached.sample() == fresh.sample(), "Snapshot logits/penalties differ");
    require(cached.prefill_cached(prompt).empty(), "Identical prompt was re-evaluated");
    // A divergence before every snapshot recomputes from an empty sequence.
    const std::vector<int32_t> early = join({prompt.begin(), prompt.begin() + length / 2}, extra);
    const auto restarted = cached.prefill_cached(early);
    fresh.reset();
    equal_steps(fresh.prefill(early), restarted);
    require(cached.sample() == fresh.sample() && cached.tokens() == early, "Restart from an empty sequence");
    // Compact (logit-free) requests keep their logits on the device for sample().
    cached.decode(5);
    require(cached.prefill_cached(early, false).empty(), "Compact snapshot restore evaluated tokens");
    require(cached.sample() == fresh.sample(), "Compact restore changed sampling");
    // Without snapshots only extensions are reused.
    options.prompt_cache_checkpoints = 0;
    tidal::Engine bare(model, options);
    bare.prefill_cached(prompt); bare.decode((extra[0] + 1) % c.vocab);
    require(bare.prefill_cached(join(prompt, extra)).size() == prompt.size() + extra.size() &&
            !bare.statistics().prompt_cache_checkpoint_bytes, "Snapshot-free rollback reused state");
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
        {
            tidal::Options options{argv[2], 0, 2};
            prompt_cache(argv[1], options);
            if (std::string(argv[2]) == "cuda") {
                options.kv_placement = tidal::KVPlacement::GPU;
                prompt_cache(argv[1], options);
            }
        }
        std::cout << "Chunked prefill/decode, rejected input, partial reset and prompt cache passed\n";
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
