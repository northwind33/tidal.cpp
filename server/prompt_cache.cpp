#include "prompt_cache.hpp"

#include <algorithm>

namespace tidal::server {
namespace {
uint64_t mix(uint64_t h, uint64_t x) {
    h ^= x + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
    h ^= h >> 31; h *= 0xBF58476D1CE4E5B9ull; h ^= h >> 29;
    return h;
}
} // namespace

std::vector<uint64_t> prefix_hashes(const std::vector<int32_t> & tokens, const std::vector<ImageSpan> & images) {
    std::vector<uint64_t> image(tokens.size(), 0);
    for (const auto & span : images) {
        const uint64_t h = span.embedding ? span.embedding->hash : span.image ? span.image->hash() : 0;
        const int count = span.embedding ? span.embedding->tokens() : -1;
        // Placeholder runs are contiguous image tokens; mark the whole run.
        for (size_t i = size_t(std::max(span.offset, 0)); i < tokens.size() && (count < 0 || int(i) < span.offset + count); ++i) {
            if (count < 0 && i > size_t(span.offset) && tokens[i] != tokens[size_t(span.offset)]) break;
            image[i] = h | 1;
        }
    }
    std::vector<uint64_t> out(tokens.size() + 1);
    out[0] = 0x6A09E667F3BCC908ull;
    for (size_t i = 0; i < tokens.size(); ++i) out[i + 1] = mix(mix(out[i], uint64_t(uint32_t(tokens[i]))), image[i]);
    return out;
}

std::vector<int> CachePlan::pins() const {
    std::vector<int> out;
    for (const auto & write : writes) out.push_back(write.position);
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

void CacheLedger::sweep(CacheClock::time_point now) {
    for (auto it = entries_.begin(); it != entries_.end();) it = it->second.expiry <= now ? entries_.erase(it) : std::next(it);
}

std::vector<int> CacheLedger::positions(size_t limit) const {
    std::vector<std::pair<CacheClock::time_point, int>> order;
    for (const auto & [key, entry] : entries_) order.emplace_back(entry.used, key.position);
    std::sort(order.begin(), order.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
    std::vector<int> out;
    for (const auto & [used, position] : order) {
        if (out.size() >= limit) break;
        if (std::find(out.begin(), out.end(), position) == out.end()) out.push_back(position);
    }
    return out;
}

void CacheLedger::evict() {
    while (entries_.size() > max_entries_) {
        auto victim = entries_.begin();
        for (auto it = entries_.begin(); it != entries_.end(); ++it)
            if (it->second.used < victim->second.used) victim = it;
        entries_.erase(victim);
    }
}

CacheUsage CacheLedger::account(const CachePlan & plan, const std::vector<uint64_t> & hashes, int reused,
                                CacheClock::time_point now) {
    CacheUsage usage;
    const int n = int(hashes.size()) - 1;
    if (!plan.enabled) { usage.read = std::clamp(reused, 0, std::max(n, 0)); return usage; }
    sweep(now);
    reused = std::clamp(reused, 0, n);
    const auto live = [&](const Key & key) -> Entry * {
        const auto it = entries_.find(key);
        return it == entries_.end() || it->second.expiry <= now ? nullptr : &it->second;
    };

    // The longest lookup boundary with a live entry that the engine reused.
    int hit = 0;
    for (const int p : plan.lookups)
        if (p > hit && p <= reused && live({plan.space, p, hashes[size_t(p)]})) hit = p;

    // Lifetime each token needs: the longest among the writes that cover it.
    std::vector<CacheWrite> writes;
    for (const auto & write : plan.writes)
        if (write.position > 0 && write.position <= n && write.ttl > 0) writes.push_back(write);
    std::sort(writes.begin(), writes.end(), [](const CacheWrite & a, const CacheWrite & b) { return a.position < b.position; });
    std::vector<int> need(writes.size()); // need[k]: tokens in (writes[k-1].position, writes[k].position]
    for (size_t k = writes.size(); k-- > 0;) need[k] = std::max(writes[k].ttl, k + 1 < writes.size() ? need[k + 1] : 0);
    const int last = writes.empty() ? 0 : writes.back().position;

    // Live entries along this prompt's prefix up to the hit, longest first.
    std::vector<std::pair<int, Entry *>> along;
    if (hit)
        for (auto & [key, entry] : entries_)
            if (key.space == plan.space && key.position <= hit && entry.expiry > now && key.hash == hashes[size_t(key.position)])
                along.emplace_back(key.position, &entry);
    std::sort(along.begin(), along.end(), [](const auto & a, const auto & b) { return a.first > b.first; });

    // Bill token ranges segment by segment: [from, to) needs `ttl`.
    size_t cover = 0;
    int covered = 0; // Longest lifetime among entries ending at or after the current token.
    const auto bill = [&](int from, int to, int ttl) {
        for (int i = to; i-- > from;) {
            // `along` is consulted from the hit downwards as i decreases.
            while (cover < along.size() && along[cover].first > i) covered = std::max(covered, along[cover++].second->ttl);
            if (i < hit && (!plan.upgrade || ttl == 0 || covered >= ttl)) ++usage.read;
            else if (ttl > 0) ++usage.written[ttl];
        }
    };
    // Segments are processed from the end of the prompt backwards so that the
    // coverage scan over `along` moves monotonically.
    if (hit > last) bill(last, hit, 0);
    for (size_t k = writes.size(); k-- > 0;) bill(k ? writes[k - 1].position : 0, writes[k].position, need[k]);

    // Reads keep the entries they used alive; writes create or extend entries.
    for (auto & [position, entry] : along) {
        const int ttl = plan.upgrade ? plan.read_ttl : entry->ttl;
        entry->expiry = std::max(entry->expiry, now + std::chrono::seconds(ttl));
        entry->used = now;
    }
    for (size_t k = 0; k < writes.size(); ++k) {
        const Key key{plan.space, writes[k].position, hashes[size_t(writes[k].position)]};
        const auto expiry = now + std::chrono::seconds(need[k]);
        if (auto * entry = live(key)) {
            entry->ttl = std::max(entry->ttl, need[k]);
            entry->expiry = std::max(entry->expiry, expiry);
            entry->used = now;
        } else entries_[key] = Entry{need[k], expiry, now};
    }
    evict();
    return usage;
}

} // namespace tidal::server
