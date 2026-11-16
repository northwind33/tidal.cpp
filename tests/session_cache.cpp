// The cache store behind Engine::prefill_cached: an engine that switches
// between sequences must resume each from stored state, and what it then
// evaluates must equal, bit for bit, a fresh engine that ran the same
// prefill/decode partition without ever leaving the sequence.
#include "tidal/tidal.hpp"

#include <algorithm>
#include <filesystem>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using Tokens = std::vector<int32_t>;
using Snapshots = std::vector<std::pair<int, bool>>;

void require(bool ok, const std::string & message) { if (!ok) throw std::runtime_error(message); }
void equal(const tidal::Step & a, const tidal::Step & b) {
    require(a.position == b.position && a.logits == b.logits && a.selections.size() == b.selections.size(),
            "State differs at position " + std::to_string(b.position));
    for (size_t i = 0; i < a.selections.size(); ++i)
        require(a.selections[i].layer == b.selections[i].layer && a.selections[i].tokens == b.selections[i].tokens, "Attention selection differs");
}
void equal_steps(const std::vector<tidal::Step> & expected, const std::vector<tidal::Step> & actual, const char * what) {
    try {
        require(expected.size() == actual.size(), "Evaluated " + std::to_string(actual.size()) + " positions instead of " + std::to_string(expected.size()));
        for (size_t i = 0; i < expected.size(); ++i) equal(expected[i], actual[i]);
    } catch (const std::exception & e) { throw std::runtime_error(std::string(what) + ": " + e.what()); }
}
// For state whose KV rows another sequence computed: a chunk of a shared prefix
// holds the rows of whichever sequence stored it first, and a CUDA batch of
// another length rounds them differently (as prefill and decode do).
void close_steps(const std::vector<tidal::Step> & expected, const std::vector<tidal::Step> & actual, const char * what) {
    require(expected.size() == actual.size(), std::string(what) + ": evaluated " + std::to_string(actual.size()) + " positions");
    for (size_t i = 0; i < expected.size(); ++i) {
        double square = 0, norm = 0;
        for (size_t j = 0; j < expected[i].logits.size(); ++j) {
            const double difference = double(expected[i].logits[j]) - actual[i].logits[j];
            square += difference * difference;
            norm += double(expected[i].logits[j]) * expected[i].logits[j];
        }
        require(expected[i].position == actual[i].position && expected[i].logits.size() == actual[i].logits.size() &&
                square <= 1e-6 * std::max(norm, 1e-20), std::string(what) + ": state differs at position " + std::to_string(expected[i].position));
    }
}
Tokens join(Tokens a, const Tokens & b) { a.insert(a.end(), b.begin(), b.end()); return a; }
int vocab = 256;
Tokens pattern(int count, int step, int start) {
    Tokens out;
    for (int i = 0; i < count; ++i) out.push_back((i * step + start) % vocab);
    return out;
}
Tokens generate(tidal::Engine & engine, int count) {
    Tokens out;
    for (int i = 0; i < count; ++i) { out.push_back(engine.sample()); engine.decode(out.back()); }
    return out;
}
std::string scratch(const std::string & name) {
    const auto path = std::filesystem::temp_directory_path() / ("tidal-session-cache-" + std::to_string(std::hash<std::string>()(name) % 100000) + "-" + name);
    std::filesystem::remove_all(path);
    return path.string();
}
tidal::Options stored(tidal::Options options, uint64_t ram, uint64_t disk = 0, const std::string & directory = {}, int branch = 0) {
    options.cache_store.ram_bytes = ram;
    options.cache_store.disk_bytes = disk;
    options.cache_store.directory = directory;
    options.cache_store.chunk_tokens = 8;
    options.cache_store.branch_tokens = branch;
    return options;
}
const uint64_t ample = 1ull << 30;

// A -> B -> A: a reply is saved when another sequence replaces it, and a
// prompt that returns evaluates only what is new.
void switching(const char * model, const tidal::Options & base, const tidal::Options & options) {
    tidal::Engine cached(model, options), fresh(model, base);
    const Tokens a = pattern(37, 7, 3), b = pattern(29, 11, 5), extra = pattern(6, 5, 1);
    require(cached.prefill_cached(a).size() == a.size(), "Cold prompt");
    const auto reply_a = generate(cached, 5);
    require(cached.prefill_cached(b).size() == b.size() && !cached.statistics().cache_store_restores, "An unknown sequence was restored");
    // A's prompt end and the end of its reply, with the chunks below them.
    require(cached.cache_store().snapshots == 2 && cached.cache_store().chunks == 5 && cached.cache_store().tokens == 40, "Saved sequence");
    const auto reply_b = generate(cached, 3);

    const auto a2 = join(join(a, reply_a), extra);
    auto reused = cached.statistics().prompt_cache_reused_tokens;
    const auto resumed = cached.prefill_cached(a2);
    require(cached.statistics().cache_store_restores == 1 && cached.statistics().cache_store_restored_tokens == 42 &&
            cached.statistics().prompt_cache_reused_tokens - reused == 42, "The reply was not resumed from the store");
    fresh.prefill(a);
    for (const auto token : reply_a) fresh.decode(token);
    equal_steps(fresh.prefill(extra), resumed, "reply resumed from the store");
    require(cached.tokens() == a2 && cached.sample() == fresh.sample(), "Sequence and sampling after a restore");

    // B returns with its reply rendered differently: it resumes at its prompt end.
    const Tokens other = pattern(6, 3, reply_b[0] + 1);
    const auto rendered = cached.prefill_cached(join(b, other));
    require(cached.statistics().cache_store_restores == 2, "The prompt end was not resumed from the store");
    fresh.reset(); fresh.prefill(b);
    equal_steps(fresh.prefill(other), rendered, "prompt end resumed from the store");

    // A's first prompt again, an earlier turn: its snapshot was superseded by
    // the later turn but is still there, and nothing is evaluated.
    require(cached.prefill_cached(a).empty() && cached.tokens() == a && cached.statistics().cache_store_restores == 3, "Earlier turn");
    fresh.reset(); fresh.prefill(a);
    require(cached.sample() == fresh.sample(), "Sampling at a restored prompt end");
    equal(fresh.decode(7), cached.decode(7));
    equal_steps(fresh.prefill(extra), cached.prefill_cached(join(join(a, {7}), extra)), "continuation of a restored earlier turn");
    const auto info = cached.cache_store();
    require(info.error.empty(), info.error);
    require(info.ram_bytes <= options.cache_store.ram_bytes || !options.cache_store.ram_bytes, "Memory budget");
    require(info.disk_bytes <= options.cache_store.disk_bytes, "Disk budget");
}

// Stored state outlives the engine: one with the same model and options
// adopts the files, one with another layout does not see them.
void restart(const char * model, const tidal::Options & base, const std::string & directory) {
    const auto options = stored(base, 16 << 10, ample, directory);
    const Tokens a = pattern(37, 7, 3), extra = pattern(6, 5, 1);
    Tokens reply;
    {
        tidal::Engine first(model, options);
        first.prefill_cached(a);
        reply = generate(first, 5);
        first.cache_flush();
        require(first.cache_store().disk_bytes > 0 && first.cache_store().snapshots == 2, "Flush");
    }
    tidal::Engine second(model, options), fresh(model, base);
    require(second.cache_store().snapshots == 2 && second.cache_store().chunks == 5, "Files of the previous engine");
    const auto resumed = second.prefill_cached(join(join(a, reply), extra));
    require(second.statistics().cache_store_restores == 1 && second.cache_store().read_bytes > 0, "No restore after a restart");
    fresh.prefill(a);
    for (const auto token : reply) fresh.decode(token);
    equal_steps(fresh.prefill(extra), resumed, "resume after a restart");
    require(second.sample() == fresh.sample(), "Sampling after a restart");
    auto other = options;
    other.cache_store.chunk_tokens = 16;
    tidal::Engine third(model, other);
    require(!third.cache_store().snapshots && third.prefill_cached(a).size() == a.size(), "Another layout saw the files");
}

// cache_sync copies ahead of time and can be interrupted between blobs.
void idle(const char * model, const tidal::Options & base) {
    tidal::Engine engine(model, stored(base, ample));
    const Tokens a = pattern(37, 7, 3), b = pattern(29, 11, 5);
    engine.prefill_cached(a);
    int polls = 0;
    require(!engine.cache_sync([&] { return ++polls > 2; }), "An interrupted copy reported completion");
    require(engine.cache_store().chunks == 2 && !engine.cache_store().snapshots, "Blobs copied before the interruption");
    require(engine.cache_sync() && engine.cache_store().chunks == 4 && engine.cache_store().snapshots == 1, "Completed copy");
    require(engine.cache_sync([] { return true; }), "Nothing was left to copy");
    // What is saved already is not copied again when the sequence is replaced.
    const auto saved = engine.statistics().cache_store_saved_bytes;
    engine.prefill_cached(b);
    require(engine.statistics().cache_store_saved_bytes == saved, "A saved sequence was copied again");
    require(engine.prefill_cached(a).empty() && engine.statistics().cache_store_restores == 1, "Restore of an idle-saved sequence");
    engine.cache_clear();
    require(!engine.cache_store().snapshots && !engine.cache_store().ram_bytes, "Clear");
    require(engine.prefill_cached(b).size() == b.size(), "A cleared store restored state");
}

// A prompt that parts from another sequence leaves a snapshot at the parting
// point; later prompts with that prefix resume there.
void branching(const char * model, const tidal::Options & base) {
    tidal::Engine engine(model, stored(base, ample, 0, {}, 8)), fresh(model, base);
    const Tokens shared = pattern(24, 7, 3), unrelated = pattern(20, 13, 9);
    const Tokens rest_a = pattern(13, 5, 40), rest_b = pattern(10, 3, 80), rest_c = pattern(9, 9, 120), rest_d = pattern(7, 11, 160);
    engine.prefill_cached(join(shared, rest_a));
    const auto b = engine.prefill_cached(join(shared, rest_b));
    require(engine.checkpoints() == Snapshots({{24, false}, {34, false}}), "Snapshot at the parting point");
    auto expected = fresh.prefill(shared);
    for (auto & step : fresh.prefill(rest_b)) expected.push_back(std::move(step));
    equal_steps(expected, b, "prompt with a parting point");
    // From the engine's own snapshot while it is there.
    const auto c = engine.prefill_cached(join(shared, rest_c));
    require(!engine.statistics().cache_store_restores, "The engine's own snapshot was not used");
    fresh.reset(); fresh.prefill(shared);
    equal_steps(fresh.prefill(rest_c), c, "parting point from the engine's snapshot");
    // From the store once an unrelated sequence has taken the engine.
    require(engine.prefill_cached(unrelated).size() == unrelated.size(), "Unrelated prompt");
    const auto d = engine.prefill_cached(join(shared, rest_d));
    require(engine.statistics().cache_store_restores == 1 && engine.statistics().cache_store_restored_tokens == 24, "Parting point from the store");
    fresh.reset(); fresh.prefill(shared);
    close_steps(fresh.prefill(rest_d), d, "parting point from the store");
    // The reply of a continuing sequence does not leave one: the prompt is kept.
    const auto reply = generate(engine, 12);
    Tokens rendered = join(join(shared, rest_d), {reply.begin(), reply.begin() + 10});
    rendered.push_back((reply[10] + 1) % vocab);
    engine.prefill_cached(rendered);
    require(engine.checkpoints() == Snapshots({{24, false}, {31, false}, {42, false}}), "A parting point inside a reply");
}

// Sequences that do not fit are evaluated again, least recently used first.
void pressure(const char * model, const tidal::Options & base) {
    const Tokens a = pattern(37, 7, 3), b = pattern(37, 11, 5), c = pattern(37, 13, 9), d = pattern(37, 17, 11);
    uint64_t one = 0;
    {
        tidal::Engine probe(model, stored(base, ample));
        probe.prefill_cached(a);
        probe.cache_sync();
        one = probe.cache_store().ram_bytes;
    }
    const uint64_t budget = 2 * one + one / 2;
    tidal::Engine engine(model, stored(base, budget)), fresh(model, base);
    engine.prefill_cached(a); engine.prefill_cached(b); engine.prefill_cached(c);
    require(engine.cache_store().snapshots == 2 && !engine.cache_store().dropped, "Two sequences fit");
    engine.prefill_cached(d);
    require(engine.cache_store().snapshots == 2 && engine.cache_store().dropped > 0 && engine.cache_store().ram_bytes <= budget, "Budget");
    const auto again = engine.prefill_cached(a);
    require(!engine.statistics().cache_store_restores, "The least recently used sequence was kept");
    equal_steps(fresh.prefill(a), again, "sequence evaluated again after eviction");
    require(engine.prefill_cached(c).empty() && engine.statistics().cache_store_restores == 1, "A recent sequence was dropped");
    // A store too small for anything behaves like none.
    tidal::Engine bare(model, stored(base, 64));
    bare.prefill_cached(a); bare.prefill_cached(b);
    fresh.reset();
    equal_steps(fresh.prefill(a), bare.prefill_cached(a), "store too small for any blob");
    require(!bare.cache_store().snapshots && !bare.statistics().cache_store_restores, "A blob beyond the whole budget was stored");
}

// reuse = false evaluates the whole prompt and still saves what it replaces.
void whole_prompt(const char * model, const tidal::Options & base) {
    tidal::Engine engine(model, stored(base, ample)), fresh(model, base);
    const Tokens a = pattern(37, 7, 3), extra = pattern(6, 5, 1);
    engine.prefill_cached(a);
    const auto reply = generate(engine, 4);
    const auto again = engine.prefill_cached(a, {}, true, {}, false);
    require(!engine.statistics().cache_store_restores && !engine.statistics().prompt_cache_reused_tokens, "reuse = false reused state");
    equal_steps(fresh.prefill(a), again, "reuse = false");
    require(engine.cache_store().snapshots == 2, "The replaced sequence was not saved");
    // The stored reply lies beyond the sequence the engine holds: it extends it.
    const auto resumed = engine.prefill_cached(join(join(a, reply), extra));
    require(engine.statistics().cache_store_restores == 1 && engine.statistics().cache_store_restored_tokens == 4, "Stored state beyond the live sequence");
    for (const auto token : reply) fresh.decode(token);
    equal_steps(fresh.prefill(extra), resumed, "stored reply beyond the live sequence");
}

// MTP rows and the pending hidden row come back with the sequence: the
// restored engine drafts what an engine that never left would draft.
void drafting(const char * model, tidal::Options base) {
    base.max_draft_tokens = 3;
    tidal::Engine cached(model, stored(base, ample)), fresh(model, base);
    const Tokens a = pattern(21, 13, 5), b = pattern(19, 7, 2);
    const auto rounds = [](tidal::Engine & engine, int count, Tokens & consumed) {
        int32_t token = engine.sample();
        for (int i = 0; i < count; ++i) {
            const auto out = engine.verify(token, engine.draft(token, 1 + i % 3));
            consumed.push_back(token);
            consumed.insert(consumed.end(), out.begin(), out.end() - 1);
            token = out.back();
        }
        return token;
    };
    Tokens consumed_cached, consumed_fresh, ignored;
    cached.prefill_cached(a, {}, false, {});
    fresh.prefill(a, false);
    const int32_t pending = rounds(cached, 6, consumed_cached);
    require(rounds(fresh, 6, consumed_fresh) == pending && consumed_cached == consumed_fresh, "Engines diverged before the switch");
    cached.prefill_cached(b, {}, false, {});
    rounds(cached, 3, ignored);
    require(cached.prefill_cached(join(a, consumed_cached), {}, false, {}).empty() && cached.statistics().cache_store_restores == 1,
            "Drafted sequence was not resumed");
    require(cached.draft(pending, 3) == fresh.draft(pending, 3), "Drafts after a restore");
    Tokens more_cached, more_fresh;
    const auto run = [&](tidal::Engine & engine, Tokens & consumed) {
        int32_t token = pending;
        for (int i = 0; i < 5; ++i) {
            const auto out = engine.verify(token, engine.draft(token, 3));
            consumed.push_back(token);
            consumed.insert(consumed.end(), out.begin(), out.end() - 1);
            token = out.back();
        }
        return token;
    };
    require(run(cached, more_cached) == run(fresh, more_fresh) && more_cached == more_fresh, "Speculative decoding after a restore");
}
} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc < 3) throw std::runtime_error("Usage: tidal-session-cache-test model.gguf cpu|cuda [mtp|SIDECAR]");
        const std::string backend = argv[2];
        tidal::Options base{backend, 0, 2};
        base.prefill_chunk_tokens = 8; base.prefill_attention_tile = 5; // Independent of free memory.
        base.sampling.repeat_penalty = 1.3f; base.sampling.penalty_last_n = 8; // Penalty history is rebuilt on a restore.
        {
            tidal::Engine probe(argv[1], base);
            vocab = probe.config().vocab;
            require(probe.config().context >= 64, "The test needs a context of at least 64 tokens");
        }
        if (argc > 3) {
            base.mtp = true;
            if (std::string(argv[3]) != "mtp") base.mtp_model = argv[3];
            drafting(argv[1], base);
            std::cout << "Session cache: MTP drafts after a restore passed\n";
            return 0;
        }
        std::vector<tidal::KVPlacement> placements{tidal::KVPlacement::CPU};
        if (backend == "cuda") placements.push_back(tidal::KVPlacement::GPU);
        for (const auto placement : placements) {
            base.kv_placement = placement;
            const std::string tag = backend + (placement == tidal::KVPlacement::GPU ? "-gpu-kv" : "-cpu-kv");
            const auto spill = scratch(tag + "-spill"), disk = scratch(tag + "-disk"), again = scratch(tag + "-restart");
            const auto scenario = [&](const char * name, const std::function<void()> & body) {
                try { body(); }
                catch (const std::exception & e) { throw std::runtime_error(tag + " " + name + ": " + e.what()); }
            };
            scenario("switching in memory", [&] { switching(argv[1], base, stored(base, ample)); });
            // Snapshots exceed this memory budget and pass straight through to disk.
            scenario("switching with spill", [&] { switching(argv[1], base, stored(base, 16 << 10, ample, spill)); });
            scenario("switching on disk", [&] { switching(argv[1], base, stored(base, 0, ample, disk)); });
            scenario("restart", [&] { restart(argv[1], base, again); });
            scenario("idle", [&] { idle(argv[1], base); });
            scenario("branching", [&] { branching(argv[1], base); });
            scenario("pressure", [&] { pressure(argv[1], base); });
            scenario("whole prompt", [&] { whole_prompt(argv[1], base); });
            for (const auto & directory : {spill, disk, again}) std::filesystem::remove_all(directory);
        }
        std::cout << "Session cache: switching, restart, idle saving, parting points, eviction and reuse = false passed\n";
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
