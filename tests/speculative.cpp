#include "tidal/tidal.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void require(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }
void close(const tidal::Step & a, const tidal::Step & b, double tolerance) {
    require(a.position == b.position && a.logits.size() == b.logits.size(), "Speculative position/logit count differs");
    double square = 0, norm = 0;
    for (size_t i = 0; i < a.logits.size(); ++i) {
        square += std::pow(double(a.logits[i]) - b.logits[i], 2); norm += double(a.logits[i]) * a.logits[i];
    }
    if (std::sqrt(square / std::max(norm, 1e-20)) > tolerance)
        throw std::runtime_error("State after verification differs at position " + std::to_string(a.position));
    require(a.selections.size() == b.selections.size(), "Selection layer count differs");
    for (size_t i = 0; i < a.selections.size(); ++i)
        require(a.selections[i].tokens == b.selections[i].tokens, "Selection after verification differs");
}
}

// Usage: tidal-speculative-test MODEL [MTP_SIDECAR|-]
// Verification with external drafts must emit exactly decode + sample's
// greedy tokens and leave the state of the accepted length, for any mix of
// right and wrong drafts. With MTP, drafts come from the model's MTP block.
int main(int argc, char ** argv) {
    try {
        if (argc < 2) throw std::runtime_error("Usage: tidal-speculative-test MODEL [mtp|SIDECAR]");
        tidal::Options options{"cuda", 0, 2};
        options.max_draft_tokens = 3;
        const bool mtp = argc > 2;
        if (mtp) { options.mtp = true; if (std::string(argv[2]) != "mtp") options.mtp_model = argv[2]; }
        tidal::Options plain = options; plain.max_draft_tokens = 0; plain.mtp = false; plain.mtp_model.clear();
        tidal::Engine reference(argv[1], plain), engine(argv[1], options);
        const auto & c = engine.config();
        std::vector<int32_t> prompt;
        for (int i = 0; i < 21; ++i) prompt.push_back((i * 13 + 5) % c.vocab);
        const int steps = std::min(40, c.context - int(prompt.size()) - 8);
        reference.prefill(prompt, false);
        std::vector<int32_t> expected;
        for (int i = 0; i < steps + 4; ++i) { expected.push_back(reference.sample()); reference.decode(expected.back(), false); }
        engine.prefill(prompt, false);
        int32_t token = engine.sample();
        require(token == expected[0], "First sample differs");
        size_t emitted = 1; // expected[0] is `token`
        int round = 0;
        while (int(emitted) < steps) {
            // Drafts: correct, then (every third round) one wrong guess.
            std::vector<int32_t> drafts;
            const int count = 1 + round % 3;
            if (mtp) drafts = engine.draft(token, count);
            else for (int i = 0; i < count; ++i) {
                int32_t guess = expected[emitted + i];
                if (round % 3 == 2 && i == count - 1) guess = (guess + 1) % c.vocab;
                if (round % 5 == 4 && i == 0) guess = (guess + 7) % c.vocab;
                drafts.push_back(guess);
            }
            const auto out = engine.verify(token, drafts);
            require(!out.empty() && out.size() <= drafts.size() + 1, "Verification emitted no or too many tokens");
            for (auto t : out) {
                if (t != expected[emitted]) {
                    std::cerr << "round " << round << " emitted " << emitted << " got " << t << " expected " << expected[emitted] << '\n';
                    throw std::runtime_error("Verification changed greedy tokens");
                }
                ++emitted;
            }
            token = out.back();
            ++round;
        }
        const auto & s = engine.statistics();
        require(s.speculative_verifications == uint64_t(round) && s.speculative_accepted <= s.speculative_drafted &&
                (mtp || (s.speculative_accepted > 0 && s.speculative_accepted < s.speculative_drafted)), "Speculative statistics");
        // The state equals the reference at the same length: compare the next
        // position's logits and selections via capture decode on both.
        tidal::Engine replay(argv[1], plain);
        replay.prefill(prompt, false);
        for (size_t i = 0; i + 1 < emitted; ++i) replay.decode(expected[i], false);
        close(replay.decode(token), engine.decode(token), .004);
        if (mtp) {
            // MTP rows written by layer-major prefill, hybrid prefill, decode and
            // verification must agree: the same sequence drafts the same tokens.
            std::vector<int32_t> sequence(prompt);
            sequence.insert(sequence.end(), expected.begin(), expected.begin() + 9);
            const int32_t next = expected[9];
            auto layer_options = options; layer_options.hybrid_batch_tokens = 4;
            tidal::Engine whole(argv[1], layer_options), mixed(argv[1], options), stepped(argv[1], options);
            whole.prefill(sequence, false);
            mixed.prefill({sequence.begin(), sequence.begin() + 25}, false);
            mixed.prefill({sequence.begin() + 25, sequence.end()}, false);
            stepped.prefill(prompt, false);
            for (size_t i = prompt.size(); i < sequence.size(); ++i) stepped.decode(sequence[i], false);
            const auto a = whole.draft(next, 3), b = mixed.draft(next, 3), d = stepped.draft(next, 3);
            std::cerr << "drafts " << a[0] << ',' << a[1] << ',' << a[2] << " | " << b[0] << ',' << b[1] << ',' << b[2]
                      << " | " << d[0] << ',' << d[1] << ',' << d[2] << '\n';
            require(a == b && b == d, "MTP rows depend on how the sequence was consumed");
            // Redrafting with another pending token must not keep the old MTP row.
            stepped.draft((next + 1) % c.vocab, 2);
            require(stepped.draft(next, 3) == d, "Redraft kept a stale MTP row");
        }
        std::cout << "verified " << s.speculative_drafted << " drafts, accepted " << s.speculative_accepted << " in " << round << " rounds\n";
        std::cout << "speculative verification passed\n";
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
