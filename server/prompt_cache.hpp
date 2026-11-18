#pragma once
// Prompt-cache accounting that follows the published semantics of the OpenAI
// and Anthropic APIs on top of the engine's physical prefix reuse.
//
// The engine reuses whatever prefix it can restore; that only changes speed.
// What a request reports as cache reads and writes is decided here, from cache
// entries that requests write at breakpoints: an entry is a prefix (identified
// by its length and a hash of its tokens and images) in a namespace, with an
// expiry. A request reads the longest prefix ending at one of its lookup
// boundaries that has a live entry and that the engine actually reused, and
// writes entries at its write breakpoints. Entries that the engine can no
// longer restore are therefore never reported as reads.
#include "tidal/tidal.hpp"

#include <chrono>
#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace tidal::server {

using CacheClock = std::chrono::steady_clock;

// prefix[i] identifies the first i positions of a prompt: token ids and, for
// image positions, the image's content hash.
std::vector<uint64_t> prefix_hashes(const std::vector<int32_t> & tokens, const std::vector<ImageSpan> & images);

struct CacheWrite {
    int position = 0;  // Prefix length in tokens.
    int ttl = 0;       // Lifetime in seconds.
};

// What a request asks of the cache, in token positions of its prompt.
struct CachePlan {
    bool enabled = false;      // Otherwise the physical reuse is reported as read.
    std::string space;         // Entries are only shared within a namespace.
    std::vector<int> lookups;  // Prefix lengths at which reads may happen.
    std::vector<CacheWrite> writes;
    // OpenAI: tokens whose entry lifetime is shorter than the request's are
    // written again (an upgrade) rather than read, and a read keeps an entry
    // alive for `read_ttl` seconds. Anthropic: any live entry is read and a
    // read renews the entry's own lifetime.
    bool upgrade = false;
    int read_ttl = 0;
    // Positions within the prompt (sorted, unique) at which the engine should
    // keep a snapshot: the write breakpoints.
    std::vector<int> pins() const;
};

struct CacheUsage {
    int read = 0;                  // Prompt tokens billed as cache reads.
    std::map<int, int> written;    // Lifetime in seconds -> prompt tokens billed as cache writes.
    int written_total() const {
        int total = 0;
        for (const auto & [ttl, count] : written) total += count;
        return total;
    }
    int written_for(int ttl) const {
        const auto it = written.find(ttl);
        return it == written.end() ? 0 : it->second;
    }
};

class CacheLedger {
public:
    explicit CacheLedger(size_t max_entries = 4096) : max_entries_(max_entries) {}
    // Accounts a request whose prompt has `hashes` (prompt length + 1 values)
    // and of which the engine reused the first `reused` positions, then
    // records its reads and writes. `now` is the request's start.
    CacheUsage account(const CachePlan & plan, const std::vector<uint64_t> & hashes, int reused, CacheClock::time_point now);
    // Drops expired entries.
    void sweep(CacheClock::time_point now);
    // Prompt positions of live entries, longest-lived first, for snapshot pinning.
    std::vector<int> positions(size_t limit) const;
    void clear() { entries_.clear(); }
    size_t size() const { return entries_.size(); }

private:
    struct Key {
        std::string space;
        int position;
        uint64_t hash;
        bool operator==(const Key & other) const {
            return position == other.position && hash == other.hash && space == other.space;
        }
    };
    struct KeyHash {
        size_t operator()(const Key & key) const {
            return std::hash<std::string>()(key.space) ^ (key.hash * 0x9E3779B97F4A7C15ull) ^ size_t(key.position);
        }
    };
    struct Entry {
        int ttl = 0;                    // Seconds of lifetime this entry was written with (the longest).
        CacheClock::time_point expiry;
        CacheClock::time_point used;    // Last write or read, for eviction.
    };
    std::unordered_map<Key, Entry, KeyHash> entries_;
    size_t max_entries_;
    void evict();
};

} // namespace tidal::server
