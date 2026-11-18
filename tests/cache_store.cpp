// The cache store on its own: blobs of synthetic sequences in a small memory
// and disk budget. Engine round trips live in tests/session_cache.cpp.
#include "cache_store.hpp"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {
namespace fs = std::filesystem;
using namespace tidal;

constexpr int chunk = 4;               // Tokens per chunk.
constexpr size_t chunk_bytes = 1000, state_bytes = 600, tail_bytes = 50;

void require(bool ok, const std::string & message) { if (!ok) throw std::runtime_error(message); }
void eventually(const std::function<bool()> & done, const char * message) {
    for (int i = 0; i < 2000; ++i) {
        if (done()) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    throw std::runtime_error(std::string("Timed out: ") + message);
}

struct Sequence {
    std::vector<int32_t> tokens;
    std::vector<uint64_t> images, chain;
};
// `length` tokens; the first `shared` are those of `base`.
Sequence sequence(int length, int seed, const Sequence * base = nullptr, int shared = 0) {
    Sequence s;
    for (int i = 0; i < length; ++i) {
        s.tokens.push_back(base && i < shared ? base->tokens[size_t(i)] : int32_t((i * 31 + seed * 97 + 5) % 1000));
        s.images.push_back(base && i < shared ? base->images[size_t(i)] : (i % 7 == 3 ? uint64_t(seed) * 1000 + 1 : 0));
    }
    s.chain = prefix_chain(s.tokens, s.images, s.tokens.size());
    return s;
}
// Contents are a function of the key, so that any copy can be checked.
Blob content(const StoreKey & key, int part, size_t size) {
    auto bytes = std::make_shared<Bytes>(size);
    uint64_t state = key.hash ^ (uint64_t(key.position) << 32) ^ uint64_t(part + 1) * 0x9E3779B97F4A7C15ull;
    for (size_t i = 0; i < size; ++i) {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        bytes->data[i] = uint8_t(state >> 56);
    }
    return bytes;
}
StoreMeta meta(const Sequence & s, StoreKind kind, int first, int position) {
    StoreMeta m;
    m.key = {kind, position, s.chain[size_t(position)]};
    m.first = first;
    m.parent = s.chain[size_t(first)];
    m.tokens.assign(s.tokens.begin() + first, s.tokens.begin() + position);
    m.images.assign(s.images.begin() + first, s.images.begin() + position);
    return m;
}
StoreKey snapshot_key(const Sequence & s, int position) { return {StoreKind::Snapshot, position, s.chain[size_t(position)]}; }
// Stores the chunks below `position` and a snapshot there, as the engine does.
bool store(CacheStore & cache, const Sequence & s, int position, SnapshotRole role = SnapshotRole::End, int anchor = -1) {
    const auto stamp = cache.tick();
    for (int k = 0; k < position / chunk; ++k) {
        const auto m = meta(s, StoreKind::Chunk, k * chunk, (k + 1) * chunk);
        if (cache.contains(m.key)) continue;
        if (!cache.put(m, {content(m.key, 0, chunk_bytes)}, stamp)) return false;
    }
    auto m = meta(s, StoreKind::Snapshot, position / chunk * chunk, position);
    m.role = role;
    m.anchor_position = anchor < 0 ? position : anchor;
    m.anchor_hash = s.chain[size_t(m.anchor_position)];
    m.mtp_next = position - 1;
    m.captured = 3;
    return cache.put(m, {content(m.key, 0, state_bytes), content(m.key, 1, tail_bytes)}, stamp, &s.chain);
}
void same(const Blob & a, const Blob & b) {
    require(a->size == b->size && std::memcmp(a->data.get(), b->data.get(), a->size) == 0, "Stored bytes differ");
}
// The position the store restores for `prompt`, loading and checking every blob; 0 for a miss.
int restore(CacheStore & cache, const Sequence & prompt, int live = 0, int floor = 0, int * chunks = nullptr) {
    const StoreQuery query{prompt.chain, prompt.tokens, prompt.images, live, floor};
    const auto match = cache.find(query, cache.tick());
    if (chunks) *chunks = match ? int(match->chunks.size()) : 0;
    if (!match) return 0;
    for (const auto & c : match->chunks) {
        cache.prefetch(c.key);
        const auto parts = cache.load(c.key);
        require(parts.size() == 1, "Chunk parts");
        same(parts[0], content(c.key, 0, chunk_bytes));
        require(c.key.position - c.first == chunk && c.key.position > live, "Chunk range");
    }
    const auto parts = cache.load(match->snapshot.key);
    require(parts.size() == 2 && match->snapshot.parts == std::vector<uint64_t>({state_bytes, tail_bytes}), "Snapshot parts");
    same(parts[0], content(match->snapshot.key, 0, state_bytes));
    same(parts[1], content(match->snapshot.key, 1, tail_bytes));
    require(match->snapshot.mtp_next == match->snapshot.key.position - 1 && match->snapshot.captured == 3, "Snapshot fields");
    cache.release(*match);
    return match->snapshot.key.position;
}
std::string scratch(const char * name) {
    const auto path = fs::temp_directory_path() / ("tidal-cache-store-" + std::to_string(std::hash<std::thread::id>()(std::this_thread::get_id())) + "-" + name);
    fs::remove_all(path);
    return path.string();
}
size_t files(const std::string & directory) {
    size_t count = 0;
    if (!fs::exists(directory)) return 0;
    for (const auto & entry : fs::recursive_directory_iterator(directory)) count += entry.is_regular_file() && entry.path().extension() == ".tkv";
    return count;
}

void memory_only() {
    CacheStore cache({1 << 20, 0, "", 7});
    const auto a = sequence(30, 1), b = sequence(27, 2, &a, 9), other = sequence(30, 3);
    require(restore(cache, a) == 0, "An empty store matched");
    require(store(cache, a, 18) && store(cache, b, 26), "Blobs were refused");
    int chunks = 0;
    require(restore(cache, a, 0, 0, &chunks) == 18 && chunks == 4, "Whole sequence from the store");
    // Rows the engine still holds are not loaded: chunks ending within them are skipped.
    require(restore(cache, a, 9, 0, &chunks) == 18 && chunks == 2, "Chunks beyond the live rows");
    require(restore(cache, a, 18, 0, &chunks) == 18 && chunks == 0, "Snapshot inside the live rows");
    require(restore(cache, a, 0, 18) == 0, "A snapshot at the floor is of no use");
    // A second sequence shares the first two chunks and has its own afterwards.
    require(restore(cache, b, 0, 0, &chunks) == 26 && chunks == 6, "Sequence that shares a prefix");
    require(cache.info().chunks == 4 + 4 && cache.info().snapshots == 2 && cache.info().tokens == 8 * chunk, "Shared chunks are stored once");
    require(restore(cache, other) == 0, "Unrelated prompt matched");
    // A prompt that continues past a snapshot restores it; one that parts before it does not.
    auto longer = sequence(30, 9, &a, 20), shorter = sequence(30, 9, &a, 17);
    require(restore(cache, longer) == 18 && restore(cache, shorter) == 0, "Prefix matching");
    // The same tokens with another image are another sequence.
    auto image = a;
    image.images[3] ^= 1;
    image.chain = prefix_chain(image.tokens, image.images, image.tokens.size());
    require(restore(cache, image) == 0, "Image content is part of the prefix");
    cache.clear();
    require(restore(cache, a) == 0 && cache.info().ram_bytes == 0 && cache.info().chunks == 0, "Clear");
}

// Later snapshots of a sequence supersede earlier ones; those go first, and
// one that is restored again becomes a branch point.
void supersede_and_evict() {
    // Room for three chunks and two snapshots.
    CacheStore cache({3 * chunk_bytes + 2 * (state_bytes + tail_bytes) + 100, 0, "", 7});
    const auto a = sequence(30, 1);
    require(store(cache, a, 6) && store(cache, a, 10), "Two turns of one sequence");
    require(cache.info().snapshots == 2, "Both snapshots fit");
    // A reply snapshot belongs to the turn it follows (its anchor).
    require(store(cache, a, 13, SnapshotRole::Generated, 10), "Reply snapshot");
    eventually([&] { return cache.info().snapshots == 2; }, "the superseded snapshot leaves first");
    require(!cache.contains(snapshot_key(a, 6)) && cache.contains(snapshot_key(a, 10)) && cache.contains(snapshot_key(a, 13)),
            "The earlier turn's snapshot was not the one dropped");
    require(cache.info().chunks == 3, "Chunks of the surviving snapshots stay");
    // The next turn supersedes both the previous end and its reply.
    require(store(cache, a, 14), "Third turn");
    eventually([&] { return cache.info().snapshots <= 2; }, "superseded snapshots give way");
    require(cache.contains(snapshot_key(a, 14)), "Latest snapshot");
    // A prompt that diverges right after a superseded snapshot restores it and keeps it.
    CacheStore roomy({1 << 20, 0, "", 7});
    require(store(roomy, a, 6) && store(roomy, a, 10) && store(roomy, a, 14), "Three turns");
    const auto fork = sequence(30, 5, &a, 11);
    require(restore(roomy, fork) == 10, "Fork after an earlier turn");
    require(store(roomy, a, 18), "Fourth turn");
    require(restore(roomy, fork) == 10, "A branch point is not superseded again");

    // Sequences compete by recency: the least recently used one goes, leaves first.
    CacheStore small({5 * chunk_bytes + 2 * (state_bytes + tail_bytes) + 100, 0, "", 7});
    const auto x = sequence(30, 11), y = sequence(30, 12, &x, 8), z = sequence(30, 13, &x, 8);
    require(store(small, x, 12) && store(small, y, 12), "Two sequences with a shared prefix");   // 4 chunks
    require(restore(small, x) == 12, "x is used again");                                         // y is now the oldest
    require(store(small, z, 16), "A third sequence needs two more chunks");                       // 6 chunks: y's own go
    require(restore(small, y) == 0, "The least recently used sequence was kept");
    require(restore(small, x) == 12 && restore(small, z) == 16, "The other sequences are whole");
    require(small.info().chunks == 5 && small.info().ram_bytes <= 5 * chunk_bytes + 2 * (state_bytes + tail_bytes) + 100, "Memory budget");
    // A sequence that cannot fit beside itself is refused rather than evicting its own chunks.
    CacheStore tiny({2 * chunk_bytes + 100, 0, "", 7});
    require(!store(tiny, x, 16) && tiny.info().ram_bytes <= 2 * chunk_bytes + 100, "Oversized sequence");
}

void disk_tier() {
    const auto directory = scratch("disk");
    const auto a = sequence(40, 21), b = sequence(40, 22, &a, 8), c = sequence(40, 23);
    const uint64_t ram = 3 * chunk_bytes, blob = 2000; // Generous bound on a file's size.
    {
        CacheStore cache({ram, 40 * blob, directory, 99});
        require(store(cache, a, 30) && store(cache, b, 22) && store(cache, c, 38), "Blobs beyond the memory budget");
        require(cache.info().ram_bytes <= ram, "Memory budget while spilling");
        eventually([&] { return cache.info().disk_bytes > 0 && cache.info().ram_bytes <= ram; }, "spill to disk");
        int chunks = 0;
        require(restore(cache, a, 0, 0, &chunks) == 30 && chunks == 7, "Restore from disk");
        require(restore(cache, b) == 22 && restore(cache, c) == 38, "Every sequence is restorable");
        require(cache.info().ram_bytes <= ram && cache.info().read_bytes > 0, "Memory budget after reading back");
        require(cache.info().error.empty(), cache.info().error);
    }
    // A new process adopts the files; another model's fingerprint sees none of them.
    require(files(directory) == 7 + 3 + 9 + 3, "Every blob was written when the store closed");
    {
        CacheStore other({ram, 40 * blob, directory, 100});
        require(restore(other, a) == 0 && other.info().chunks == 0, "Fingerprints are separate");
#if !defined(_WIN32)
        bool refused = false;
        try { CacheStore second({ram, 40 * blob, directory, 100}); } catch (const std::runtime_error &) { refused = true; }
        require(refused, "Two stores shared a directory");
#endif
    }
    {
        CacheStore cache({ram, 40 * blob, directory, 99});
        require(cache.info().snapshots == 3 && cache.info().chunks == 7 + 3 + 9, "Files of the previous run");
        require(restore(cache, a) == 30 && restore(cache, b) == 22 && restore(cache, c) == 38, "Restore after a restart");
    }
    {
        // A damaged file is a miss, not wrong state: sequences that need one cannot be restored.
        size_t damaged = 0;
        for (const auto & entry : fs::recursive_directory_iterator(directory))
            if (entry.is_regular_file() && entry.path().filename().string().rfind("c-", 0) == 0 &&
                entry.path().filename().string().find("-20.tkv") != std::string::npos) {
                std::fstream file(entry.path(), std::ios::in | std::ios::out | std::ios::binary);
                file.seekp(-10, std::ios::end);
                file.put('\x5a'); file.put('\x5a');
                ++damaged;
            }
        require(damaged == 3, "Chunk files to damage");
        CacheStore cache({ram, 40 * blob, directory, 99});
        const StoreQuery query{a.chain, a.tokens, a.images, 0, 0};
        const auto match = cache.find(query, cache.tick());
        require(match.has_value(), "The damage is not visible in the header");
        bool failed = false;
        try { for (const auto & c2 : match->chunks) cache.load(c2.key); } catch (const std::runtime_error &) { failed = true; }
        cache.release(*match);
        require(failed && !cache.info().error.empty(), "A damaged chunk was loaded");
        require(restore(cache, a) == 0, "A sequence with a damaged chunk matched again");
    }
    {
        CacheStore cache({ram, 40 * blob, directory, 99});
        cache.clear();
        require(files(directory) == 0, "Clear removes the files");
    }
    fs::remove_all(directory);

    // The disk budget bounds the files; what fits in neither tier is dropped, oldest sequence first.
    const auto bounded = scratch("bounded");
    {
        const uint64_t disk = 6 * 1400;
        CacheStore cache({2 * chunk_bytes, disk, bounded, 5});
        const auto p = sequence(40, 31), q = sequence(40, 32), r = sequence(40, 33);
        require(store(cache, p, 12) && store(cache, q, 12) && store(cache, r, 12), "Sequences beyond both budgets");
        cache.flush();
        const auto info = cache.info();
        require(info.disk_bytes <= disk && info.ram_bytes <= 2 * chunk_bytes, "Budgets");
        require(info.dropped > 0 && restore(cache, p) == 0, "The oldest sequence was kept");
        require(restore(cache, r) == 12, "The newest sequence was dropped");
        uint64_t on_disk = 0;
        for (const auto & entry : fs::recursive_directory_iterator(bounded)) if (entry.is_regular_file()) on_disk += entry.file_size();
        require(on_disk <= disk, "Files exceed the disk budget");
    }
    fs::remove_all(bounded);

    // Disk only: blobs pass through memory one at a time.
    const auto direct = scratch("direct");
    {
        CacheStore cache({0, 40 * blob, direct, 5});
        require(store(cache, a, 30), "Disk-only store");
        eventually([&] { return cache.info().ram_bytes == 0; }, "blobs leave memory");
        require(restore(cache, a) == 30 && cache.info().ram_bytes == 0, "Restore from a disk-only store");
    }
    fs::remove_all(direct);
}
} // namespace

int main() {
    try {
        memory_only();
        supersede_and_evict();
        disk_tier();
        std::cout << "Cache store: sharing, eviction order, superseding, spilling, restart and damage passed\n";
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
