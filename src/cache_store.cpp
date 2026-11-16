#include "cache_store.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace tidal {
namespace {
namespace fs = std::filesystem;

constexpr char magic[8] = {'T', 'I', 'D', 'A', 'L', 'K', 'V', '1'};
constexpr uint32_t file_version = 1;
constexpr const char * lock_name = ".lock";
constexpr uint64_t max_header_bytes = 64ull << 20;

inline uint64_t rotl(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }
uint64_t mix(uint64_t h, uint64_t x) {
    h ^= x + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
    h ^= h >> 31; h *= 0xBF58476D1CE4E5B9ull; h ^= h >> 29;
    return h;
}

struct Writer {
    std::vector<uint8_t> bytes;
    template<class T> void put(const T & value) {
        const auto * p = reinterpret_cast<const uint8_t *>(&value);
        bytes.insert(bytes.end(), p, p + sizeof(T));
    }
    template<class T> void put(const std::vector<T> & values) {
        const auto * p = reinterpret_cast<const uint8_t *>(values.data());
        bytes.insert(bytes.end(), p, p + values.size() * sizeof(T));
    }
};
struct Reader {
    const uint8_t * data; size_t size, at = 0;
    template<class T> bool get(T & value) {
        if (size - at < sizeof(T)) return false;
        std::memcpy(&value, data + at, sizeof(T)); at += sizeof(T);
        return true;
    }
    template<class T> bool get(std::vector<T> & values, uint64_t count) {
        if ((size - at) / sizeof(T) < count) return false;
        values.resize(size_t(count));
        if (count) std::memcpy(values.data(), data + at, size_t(count) * sizeof(T));
        at += size_t(count) * sizeof(T);
        return true;
    }
};

size_t header_size(const StoreMeta & meta) {
    // Magic, size, version, fingerprint and payload checksum; 72 bytes of fixed fields; the header checksum.
    return 8 + 4 + 4 + 8 + 8 + 72 + 8 +
           meta.tokens.size() * sizeof(int32_t) + meta.images.size() * sizeof(uint64_t) + meta.parts.size() * sizeof(uint64_t);
}
std::vector<uint8_t> encode_header(const StoreMeta & meta, uint64_t fingerprint, uint64_t checksum) {
    Writer w;
    w.bytes.insert(w.bytes.end(), magic, magic + 8);
    w.put(uint32_t(header_size(meta))); w.put(file_version); w.put(fingerprint); w.put(checksum);
    w.put(uint32_t(meta.key.kind)); w.put(uint32_t(meta.role)); w.put(meta.key.hash); w.put(meta.key.position); w.put(meta.first);
    w.put(meta.parent); w.put(meta.anchor_hash); w.put(meta.anchor_position); w.put(meta.mtp_next); w.put(meta.captured);
    w.put(uint32_t(meta.tokens.size())); w.put(uint32_t(meta.images.size()));
    w.put(uint32_t(meta.parts.size())); w.put(uint32_t(0));
    w.put(meta.tokens); w.put(meta.images); w.put(meta.parts);
    w.put(hash_bytes(w.bytes.data(), w.bytes.size()));
    if (w.bytes.size() != header_size(meta)) throw std::logic_error("Cache store header layout");
    return w.bytes;
}
bool decode_header(const uint8_t * data, size_t size, StoreMeta & meta, uint64_t & fingerprint, uint64_t & checksum) {
    if (size < 8 + 4 + 8 || std::memcmp(data, magic, 8) != 0) return false;
    uint64_t stored = 0;
    std::memcpy(&stored, data + size - 8, 8);
    if (stored != hash_bytes(data, size - 8)) return false;
    Reader r{data, size - 8, 8};
    uint32_t bytes = 0, version = 0, kind = 0, role = 0, tokens = 0, images = 0, parts = 0, pad = 0;
    if (!r.get(bytes) || !r.get(version) || bytes != size || version != file_version) return false;
    if (!r.get(fingerprint) || !r.get(checksum) || !r.get(kind) || !r.get(role) || !r.get(meta.key.hash) ||
        !r.get(meta.key.position) || !r.get(meta.first) || !r.get(meta.parent) || !r.get(meta.anchor_hash) ||
        !r.get(meta.anchor_position) || !r.get(meta.mtp_next) || !r.get(meta.captured) || !r.get(tokens) || !r.get(images) ||
        !r.get(parts) || !r.get(pad)) return false;
    if (kind > 1 || role > 2) return false;
    meta.key.kind = StoreKind(kind); meta.role = SnapshotRole(role);
    if (!r.get(meta.tokens, tokens) || !r.get(meta.images, images) || !r.get(meta.parts, parts) || r.at != r.size) return false;
    return meta.first >= 0 && meta.key.position >= meta.first && meta.tokens.size() == size_t(meta.key.position - meta.first) &&
           (meta.images.empty() || meta.images.size() == meta.tokens.size());
}
uint64_t payload_bytes(const StoreMeta & meta) {
    uint64_t total = 0;
    for (const auto part : meta.parts) total += part;
    return total;
}
uint64_t payload_checksum(const std::vector<Blob> & parts, uint64_t fingerprint) {
    uint64_t h = fingerprint;
    for (const auto & part : parts) h = hash_bytes(part->data.get(), part->size, h);
    return h;
}
[[noreturn]] void fail(const std::string & what, const std::string & path) {
    throw std::runtime_error("Cache store: " + what + " " + path + (errno ? std::string(": ") + std::strerror(errno) : ""));
}
void write_file(const std::string & path, const StoreMeta & meta, const std::vector<Blob> & parts, uint64_t fingerprint) {
    const auto header = encode_header(meta, fingerprint, payload_checksum(parts, fingerprint));
    const std::string temporary = path + ".tmp";
    errno = 0;
    {
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        if (!out) fail("cannot create", temporary);
        out.write(reinterpret_cast<const char *>(header.data()), std::streamsize(header.size()));
        for (const auto & part : parts) out.write(reinterpret_cast<const char *>(part->data.get()), std::streamsize(part->size));
        out.close();
        if (!out) {
            const int error = errno;
            std::error_code ignored;
            fs::remove(temporary, ignored);
            errno = error;
            fail("cannot write", temporary);
        }
    }
    std::error_code error;
    fs::rename(temporary, path, error);
    if (error) {
        std::error_code ignored;
        fs::remove(temporary, ignored);
        throw std::runtime_error("Cache store: cannot rename " + temporary + ": " + error.message());
    }
}
std::vector<Blob> read_file(const std::string & path, const StoreMeta & meta, uint64_t fingerprint) {
    errno = 0;
    std::ifstream in(path, std::ios::binary);
    if (!in) fail("cannot open", path);
    std::vector<uint8_t> header(header_size(meta));
    in.read(reinterpret_cast<char *>(header.data()), std::streamsize(header.size()));
    StoreMeta stored;
    uint64_t stored_fingerprint = 0, checksum = 0;
    if (!in || !decode_header(header.data(), header.size(), stored, stored_fingerprint, checksum) || stored_fingerprint != fingerprint ||
        !(stored.key == meta.key) || stored.parts != meta.parts)
        throw std::runtime_error("Cache store: damaged header in " + path);
    std::vector<Blob> parts;
    for (const auto size : meta.parts) {
        auto part = std::make_shared<Bytes>(size_t(size));
        in.read(reinterpret_cast<char *>(part->data.get()), std::streamsize(size));
        if (!in) throw std::runtime_error("Cache store: truncated file " + path);
        parts.push_back(std::move(part));
    }
    if (payload_checksum(parts, fingerprint) != checksum) throw std::runtime_error("Cache store: checksum mismatch in " + path);
    return parts;
}
std::string hex(uint64_t value, int digits) {
    static const char * alphabet = "0123456789abcdef";
    std::string out(size_t(digits), '0');
    for (int i = digits - 1; i >= 0; --i, value >>= 4) out[size_t(i)] = alphabet[value & 15];
    return out;
}
} // namespace

uint64_t hash_bytes(const void * data, size_t size, uint64_t seed) {
    constexpr uint64_t p1 = 0x9E3779B185EBCA87ull, p2 = 0xC2B2AE3D27D4EB4Full, p3 = 0x165667B19E3779F9ull;
    const auto * p = static_cast<const uint8_t *>(data);
    uint64_t lane[4] = {seed + p1 + p2, seed + p2, seed, seed - p1};
    size_t i = 0;
    for (; i + 32 <= size; i += 32)
        for (int l = 0; l < 4; ++l) {
            uint64_t x;
            std::memcpy(&x, p + i + 8 * l, 8);
            lane[l] = rotl(lane[l] + x * p2, 31) * p1;
        }
    uint64_t h = rotl(lane[0], 1) + rotl(lane[1], 7) + rotl(lane[2], 12) + rotl(lane[3], 18) + size;
    for (; i + 8 <= size; i += 8) {
        uint64_t x;
        std::memcpy(&x, p + i, 8);
        h = rotl(h ^ (rotl(x * p2, 31) * p1), 27) * p1 + p3;
    }
    for (; i < size; ++i) h = rotl(h ^ (p[i] * p3), 11) * p1;
    h ^= h >> 33; h *= p2; h ^= h >> 29; h *= p3; h ^= h >> 32;
    return h;
}

std::vector<uint64_t> prefix_chain(const std::vector<int32_t> & tokens, const std::vector<uint64_t> & images, size_t count) {
    count = std::min(count, tokens.size());
    std::vector<uint64_t> out(count + 1);
    out[0] = 0x6A09E667F3BCC908ull;
    for (size_t i = 0; i < count; ++i)
        out[i + 1] = mix(mix(out[i], uint64_t(uint32_t(tokens[i]))), i < images.size() ? images[i] : 0);
    return out;
}

struct CacheStore::Entry {
    StoreMeta meta;
    uint64_t bytes = 0, file_bytes = 0;  // Payload; payload plus header.
    std::vector<Blob> ram, staged;       // Resident parts; parts read ahead and not yet claimed.
    bool disk = false, writing = false, reading = false;
    bool superseded = false;             // A snapshot its sequence has moved past: the first to go, never written.
    bool oversize = false;               // Larger than the whole memory budget: passes straight through to disk.
    int pins = 0, children = 0;
    uint64_t stamp = 0;
    bool leaf() const { return meta.key.kind == StoreKind::Snapshot || children == 0; }
    bool held() const { return !ram.empty() || disk; }
};

// Coldest first. At equal stamps snapshots go before chunks and deeper chunks
// before shallower ones, so that dependents leave before what they need.
bool CacheStore::Rank::operator()(const Entry * a, const Entry * b) const {
    if (a->superseded != b->superseded) return a->superseded;
    if (a->stamp != b->stamp) return a->stamp < b->stamp;
    if (a->meta.key.kind != b->meta.key.kind) return a->meta.key.kind == StoreKind::Snapshot;
    if (a->meta.key.position != b->meta.key.position) return a->meta.key.position > b->meta.key.position;
    if (a->meta.key.hash != b->meta.key.hash) return a->meta.key.hash < b->meta.key.hash;
    return a < b;
}
bool CacheStore::RamRank::operator()(const Entry * a, const Entry * b) const {
    if (a->oversize != b->oversize) return a->oversize;
    return Rank()(a, b);
}

CacheStore::CacheStore(const Config & config) : config_(config) {
    if (config_.disk_bytes) {
        if (config_.directory.empty()) throw std::invalid_argument("The cache store's disk budget requires a directory");
        root_ = (fs::path(config_.directory) / hex(config_.fingerprint, 16)).string();
        std::error_code error;
        fs::create_directories(root_, error);
        if (error) throw std::runtime_error("Cannot create cache directory " + root_ + ": " + error.message());
#if !defined(_WIN32)
        // Two processes would each fill the budget and delete each other's files.
        lock_ = ::open((root_ + "/" + lock_name).c_str(), O_RDWR | O_CREAT, 0644);
        if (lock_ < 0 || ::flock(lock_, LOCK_EX | LOCK_NB) != 0) {
            if (lock_ >= 0) ::close(lock_);
            throw std::runtime_error("The cache directory " + root_ + " is in use by another process");
        }
#endif
        try { scan(); }
        catch (...) {
#if !defined(_WIN32)
            ::close(lock_);
#endif
            throw;
        }
    }
    spiller_ = std::thread([this] { spill(); });
    if (config_.disk_bytes)
        for (int i = 0; i < 2; ++i) readers_.emplace_back([this] { read_loop(); });
}

CacheStore::~CacheStore() {
    try { flush(); } catch (...) {}
    {
        std::lock_guard lock(mutex_);
        stop_ = true;
    }
    work_.notify_all();
    changed_.notify_all();
    spiller_.join();
    for (auto & reader : readers_) reader.join();
#if !defined(_WIN32)
    if (lock_ >= 0) ::close(lock_);
#endif
}

std::string CacheStore::path_of(const StoreKey & key) const {
    return root_ + "/" + hex(key.hash & 0xff, 2) + "/" + (key.kind == StoreKind::Chunk ? "c-" : "s-") + hex(key.hash, 16) + "-" +
           std::to_string(key.position) + ".tkv";
}

// Adopts the files of an earlier run. Their order of use is taken from the
// modification times; files that are damaged or that nothing can reach go.
void CacheStore::scan() {
    struct Found { fs::file_time_type time; std::unique_ptr<Entry> entry; };
    std::vector<Found> found;
    std::error_code error;
    for (fs::recursive_directory_iterator it(root_, error), end; !error && it != end; it.increment(error)) {
        if (!it->is_regular_file(error)) continue;
        const auto path = it->path();
        if (path.filename() == lock_name) continue;
        bool valid = false;
        if (path.extension() == ".tkv") {
            std::ifstream in(path, std::ios::binary);
            uint8_t head[12];
            uint32_t size = 0;
            if (in.read(reinterpret_cast<char *>(head), sizeof head)) std::memcpy(&size, head + 8, 4);
            if (size >= sizeof head && size <= max_header_bytes) {
                std::vector<uint8_t> header(size);
                std::memcpy(header.data(), head, sizeof head);
                auto entry = std::make_unique<Entry>();
                uint64_t fingerprint = 0, checksum = 0;
                std::error_code ignored;
                if (in.read(reinterpret_cast<char *>(header.data() + sizeof head), std::streamsize(size - sizeof head)) &&
                    decode_header(header.data(), header.size(), entry->meta, fingerprint, checksum) && fingerprint == config_.fingerprint &&
                    fs::file_size(path, ignored) == size + payload_bytes(entry->meta) &&
                    fs::path(path_of(entry->meta.key)).lexically_normal() == path.lexically_normal()) {
                    entry->bytes = payload_bytes(entry->meta);
                    entry->file_bytes = entry->bytes + size;
                    entry->disk = true;
                    entry->oversize = entry->bytes > config_.ram_bytes;
                    found.push_back({fs::last_write_time(path, ignored), std::move(entry)});
                    valid = true;
                }
            }
        }
        if (!valid) { std::error_code ignored; fs::remove(path, ignored); }
    }
    std::sort(found.begin(), found.end(), [](const Found & a, const Found & b) { return a.time < b.time; });
    for (auto & item : found) {
        auto & entry = *item.entry;
        entry.stamp = ++clock_;
        const auto key = entry.meta.key;
        if (index_.contains(key)) continue;
        disk_used_ += entry.file_bytes;
        if (key.kind == StoreKind::Snapshot) { snapshots_.insert(&entry); ++totals_.snapshots; }
        else { ++totals_.chunks; totals_.tokens += entry.meta.tokens.size(); }
        index_.emplace(key, std::move(item.entry));
    }
    std::vector<Entry *> order;
    for (auto & [key, entry] : index_) order.push_back(entry.get());
    for (auto * entry : order)
        if (auto * parent = parent_of(*entry)) ++parent->children;
    for (auto * entry : order) link(*entry);
    ++clock_;
    // Blobs whose chain to the start is broken, and chunks without a snapshot
    // below them, cannot serve a restore.
    std::sort(order.begin(), order.end(), [](const Entry * a, const Entry * b) { return a->meta.key.position < b->meta.key.position; });
    std::vector<StoreKey> broken;
    std::set<Entry *> gone;
    for (auto * entry : order)
        if (entry->meta.first > 0) {
            auto * parent = parent_of(*entry);
            if (!parent || gone.contains(parent)) { gone.insert(entry); broken.push_back(entry->meta.key); }
        }
    for (auto it = broken.rbegin(); it != broken.rend(); ++it)
        if (auto * entry = get(*it); entry && entry->leaf()) erase(*entry);
    for (;;) {
        Entry * loose = nullptr;
        for (auto * entry : leaf_order_)
            if (entry->meta.key.kind == StoreKind::Chunk) { loose = entry; break; }
        if (!loose) break;
        erase(*loose);
    }
    // A chunk is at least as recent as the snapshots that need it.
    std::vector<Entry *> snapshots(snapshots_.begin(), snapshots_.end());
    for (auto * snapshot : snapshots)
        if (auto * parent = parent_of(*snapshot)) touch_chain(*parent, snapshot->stamp);
    while (disk_used_ > config_.disk_bytes) {
        auto * victim = coldest_leaf();
        if (!victim) break;
        erase(*victim);
    }
    totals_.dropped = 0; totals_.dropped_bytes = 0;
}

CacheStore::Entry * CacheStore::get(const StoreKey & key) const {
    const auto it = index_.find(key);
    return it == index_.end() ? nullptr : it->second.get();
}
CacheStore::Entry * CacheStore::parent_of(const Entry & entry) const {
    return entry.meta.first > 0 ? get({StoreKind::Chunk, entry.meta.first, entry.meta.parent}) : nullptr;
}
// Ranks are read by the ordered sets: unlink before changing one, link after.
void CacheStore::link(Entry & entry) {
    if (!entry.ram.empty()) ram_order_.insert(&entry);
    if (entry.leaf()) leaf_order_.insert(&entry);
}
void CacheStore::unlink(Entry & entry) {
    ram_order_.erase(&entry);
    leaf_order_.erase(&entry);
}
void CacheStore::touch_chain(Entry & entry, uint64_t stamp) {
    for (Entry * e = &entry; e && e->stamp < stamp; e = parent_of(*e)) {
        unlink(*e);
        e->stamp = stamp;
        link(*e);
    }
}
void CacheStore::set_ram(Entry & entry, std::vector<Blob> parts) {
    unlink(entry);
    entry.ram = std::move(parts);
    ram_used_ += entry.bytes;
    if (entry.oversize) ++oversize_;
    link(entry);
    if (entry.disk) shared_.insert(&entry);
}
void CacheStore::drop_ram(Entry & entry) {
    unlink(entry);
    entry.ram.clear();
    ram_used_ -= entry.bytes;
    if (entry.oversize) --oversize_;
    link(entry);
    shared_.erase(&entry);
}
void CacheStore::drop_disk(Entry & entry) {
    std::error_code ignored;
    fs::remove(path_of(entry.meta.key), ignored);
    disk_used_ -= entry.file_bytes;
    entry.disk = false;
    shared_.erase(&entry);
}
bool CacheStore::deletable(const Entry & entry) const {
    return entry.pins == 0 && !entry.reading && !entry.writing && entry.leaf();
}
// Removes a leaf and, with it, the chunks that only it needed. Chunks of the
// operation in progress stay: their snapshot may still be on its way.
void CacheStore::erase(Entry & first) {
    for (Entry * entry = &first; entry;) {
        Entry * parent = parent_of(*entry);
        unlink(*entry);
        if (!entry->ram.empty()) {
            ram_used_ -= entry->bytes;
            if (entry->oversize) --oversize_;
        }
        if (entry->disk) {
            std::error_code ignored;
            fs::remove(path_of(entry->meta.key), ignored);
            disk_used_ -= entry->file_bytes;
        }
        shared_.erase(entry);
        if (entry->meta.key.kind == StoreKind::Snapshot) { snapshots_.erase(entry); --totals_.snapshots; }
        else { --totals_.chunks; totals_.tokens -= entry->meta.tokens.size(); }
        ++totals_.dropped;
        totals_.dropped_bytes += entry->bytes;
        const StoreKey key = entry->meta.key;
        index_.erase(key);
        entry = nullptr;
        if (parent) {
            unlink(*parent);
            --parent->children;
            link(*parent);
            if (parent->children == 0 && parent->stamp < clock_ && deletable(*parent)) entry = parent;
        }
    }
}
void CacheStore::supersede(Entry & entry) {
    if (entry.superseded || entry.meta.role == SnapshotRole::Break) return;
    unlink(entry);
    entry.superseded = true;
    link(entry);
}
// Keeps parts that came from disk in memory when blobs that are colder and
// already on disk can make the room.
bool CacheStore::promote(Entry & entry, const std::vector<Blob> & parts) {
    if (entry.oversize || !entry.ram.empty()) return false;
    while (ram_used_ + entry.bytes > config_.ram_bytes) {
        if (ram_order_.empty()) return false;
        Entry * victim = *ram_order_.begin();
        if (!victim->disk || victim->writing || !Rank()(victim, &entry)) return false;
        drop_ram(*victim);
    }
    set_ram(entry, parts);
    return true;
}
// The coldest part of host memory that is kept written to disk ahead of
// need, so that it can be dropped at once: room for a snapshot and a few
// chunks, which is what a prompt that replaces a sequence saves on the spot.
uint64_t CacheStore::headroom() const {
    if (!(config_.disk_bytes && disk_ok_)) return 0;
    return std::min(config_.ram_bytes / 2, std::max(std::min<uint64_t>(config_.ram_bytes / 8, 256ull << 20), 2 * largest_));
}
CacheStore::Entry * CacheStore::coldest_leaf() const {
    for (auto * entry : leaf_order_)
        if (deletable(*entry)) return entry;
    return nullptr;
}

uint64_t CacheStore::tick() {
    std::lock_guard lock(mutex_);
    return ++clock_;
}
bool CacheStore::contains(const StoreKey & key) const {
    std::lock_guard lock(mutex_);
    const auto * entry = get(key);
    return entry && entry->held();
}

bool CacheStore::put(StoreMeta meta, std::vector<Blob> parts, uint64_t stamp, const std::vector<uint64_t> * path) {
    uint64_t bytes = 0;
    meta.parts.clear();
    for (const auto & part : parts) { meta.parts.push_back(part->size); bytes += part->size; }
    std::unique_lock lock(mutex_);
    const auto key = meta.key;
    const auto parent = [&]() -> Entry * { return get({StoreKind::Chunk, meta.first, meta.parent}); };
    const auto present = [&] {
        const auto * entry = get(key);
        return entry && entry->held();
    };
    if (present()) return true;
    if (get(key) || (meta.first > 0 && !parent())) return false;
    const bool oversize = bytes > config_.ram_bytes;
    if (oversize && !(config_.disk_bytes && disk_ok_)) return false;
    if (oversize ? oversize_ > 0 : ram_used_ + bytes > config_.ram_bytes) {
        Waiter waiter{bytes, stamp};
        waiters_.push_back(&waiter);
        wake();
        changed_.wait(lock, [&] { return waiter.done; });
        if (!waiter.admitted) return false;
        if (present()) return true;
        if (get(key) || (meta.first > 0 && !parent())) return false;
    }
    auto owned = std::make_unique<Entry>();
    Entry & entry = *owned;
    entry.meta = std::move(meta);
    entry.bytes = bytes;
    entry.file_bytes = bytes + header_size(entry.meta);
    entry.stamp = stamp;
    entry.oversize = oversize;
    if (!oversize) largest_ = std::max(largest_, bytes);
    index_.emplace(key, std::move(owned));
    Entry * above = parent_of(entry);
    if (above) {
        unlink(*above);
        ++above->children;
        link(*above);
    }
    set_ram(entry, std::move(parts));
    ++totals_.stored;
    if (key.kind == StoreKind::Chunk) { ++totals_.chunks; totals_.tokens += entry.meta.tokens.size(); }
    else {
        snapshots_.insert(&entry);
        ++totals_.snapshots;
        if (above) touch_chain(*above, stamp);
        if (entry.meta.role == SnapshotRole::End && path) {
            std::vector<Entry *> passed;
            for (auto * other : snapshots_)
                if (other != &entry && other->meta.anchor_position < key.position && size_t(other->meta.anchor_position) < path->size() &&
                    (*path)[size_t(other->meta.anchor_position)] == other->meta.anchor_hash)
                    passed.push_back(other);
            for (auto * other : passed) supersede(*other);
        }
    }
    wake();
    return true;
}

std::optional<StoreMatch> CacheStore::find(const StoreQuery & query, uint64_t stamp) {
    std::lock_guard lock(mutex_);
    const int n = int(query.tokens.size());
    std::vector<Entry *> candidates;
    for (auto * snapshot : snapshots_) {
        const int position = snapshot->meta.key.position;
        if (position > query.floor && position <= n && query.prefix[size_t(position)] == snapshot->meta.key.hash && snapshot->held())
            candidates.push_back(snapshot);
    }
    std::sort(candidates.begin(), candidates.end(), [](const Entry * a, const Entry * b) { return a->meta.key.position > b->meta.key.position; });
    // Tokens and images of the blob's rows equal the prompt's.
    const auto same = [&](const StoreMeta & meta) {
        if (meta.key.position > n || query.prefix[size_t(meta.key.position)] != meta.key.hash) return false;
        if (!std::equal(meta.tokens.begin(), meta.tokens.end(), query.tokens.begin() + meta.first)) return false;
        for (size_t i = 0; i < meta.tokens.size(); ++i)
            if ((meta.images.empty() ? 0 : meta.images[i]) != query.images[size_t(meta.first) + i]) return false;
        return true;
    };
    for (auto * snapshot : candidates) {
        if (!same(snapshot->meta)) continue;
        std::vector<Entry *> chain;
        bool complete = true;
        // A chunk that ends within the live rows, and all before it, need not be stored.
        for (const Entry * cursor = snapshot; cursor->meta.first > query.live;) {
            Entry * parent = parent_of(*cursor);
            if (!parent || !parent->held() || !same(parent->meta)) { complete = false; break; }
            chain.push_back(parent);
            cursor = parent;
        }
        if (!complete) continue;
        StoreMatch match;
        if (snapshot->superseded || snapshot->stamp < stamp) {
            // A snapshot that a sequence returns to after moving past it is a branch point.
            unlink(*snapshot);
            if (snapshot->superseded) { snapshot->superseded = false; snapshot->meta.role = SnapshotRole::Break; }
            snapshot->stamp = std::max(snapshot->stamp, stamp);
            link(*snapshot);
        }
        if (auto * parent = parent_of(*snapshot)) touch_chain(*parent, stamp);
        match.snapshot = snapshot->meta;
        ++snapshot->pins;
        for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
            match.chunks.push_back((*it)->meta);
            ++(*it)->pins;
        }
        return match;
    }
    return std::nullopt;
}

void CacheStore::release(const StoreMatch & match) {
    std::lock_guard lock(mutex_);
    const auto unpin = [&](const StoreKey & key) {
        auto * entry = get(key);
        if (!entry) return;
        --entry->pins;
        entry->staged.clear();
        if (!entry->held() && deletable(*entry)) erase(*entry);
    };
    unpin(match.snapshot.key);
    for (const auto & chunk : match.chunks) unpin(chunk.key);
    wake();
}

void CacheStore::used(const StoreKey & key, uint64_t stamp) {
    std::lock_guard lock(mutex_);
    auto * entry = get(key);
    if (!entry || key.kind != StoreKind::Snapshot) return;
    unlink(*entry);
    if (entry->superseded) { entry->superseded = false; entry->meta.role = SnapshotRole::Break; }
    entry->stamp = std::max(entry->stamp, stamp);
    link(*entry);
    if (auto * parent = parent_of(*entry)) touch_chain(*parent, stamp);
}

// Reads the entry's file without the lock. The entry cannot be erased
// meanwhile; a file that fails to read is given up.
std::vector<Blob> CacheStore::read(std::unique_lock<std::mutex> & lock, Entry & entry) {
    entry.reading = true;
    ++busy_;
    const auto path = path_of(entry.meta.key);
    StoreMeta meta;
    meta.key = entry.meta.key; meta.first = entry.meta.first; meta.tokens = entry.meta.tokens; meta.images = entry.meta.images;
    meta.parts = entry.meta.parts;
    lock.unlock();
    std::vector<Blob> parts;
    std::string error;
    try { parts = read_file(path, meta, config_.fingerprint); }
    catch (const std::exception & e) { error = e.what(); }
    lock.lock();
    entry.reading = false;
    --busy_;
    changed_.notify_all();
    if (!error.empty()) {
        totals_.error = error;
        drop_disk(entry);
        if (!entry.held() && deletable(entry)) erase(entry);
        throw std::runtime_error(error);
    }
    totals_.read_bytes += entry.file_bytes;
    return parts;
}

std::vector<Blob> CacheStore::load(const StoreKey & key) {
    std::unique_lock lock(mutex_);
    for (;;) {
        Entry * entry = get(key);
        if (!entry) throw std::runtime_error("Cache store: the state is no longer stored");
        if (!entry->ram.empty()) { ++totals_.loaded; return entry->ram; }
        if (!entry->staged.empty()) {
            auto parts = std::move(entry->staged);
            entry->staged.clear();
            promote(*entry, parts);
            ++totals_.loaded;
            return parts;
        }
        if (entry->reading) { changed_.wait(lock); continue; }
        if (!entry->disk) throw std::runtime_error("Cache store: the state is no longer stored");
        auto parts = read(lock, *entry);
        promote(*entry, parts);
        ++totals_.loaded;
        return parts;
    }
}

void CacheStore::prefetch(const StoreKey & key) {
    std::lock_guard lock(mutex_);
    const auto * entry = get(key);
    if (!entry || !entry->ram.empty() || !entry->staged.empty() || entry->reading || !entry->disk) return;
    read_queue_.push_back(key);
    work_.notify_all();
}

void CacheStore::read_loop() {
    std::unique_lock lock(mutex_);
    for (;;) {
        work_.wait(lock, [&] { return stop_ || !read_queue_.empty(); });
        if (stop_) return;
        const auto key = read_queue_.front();
        read_queue_.pop_front();
        Entry * entry = get(key);
        if (!entry || !entry->ram.empty() || !entry->staged.empty() || entry->reading || !entry->disk) continue;
        try {
            auto parts = read(lock, *entry);
            if (!promote(*entry, parts)) entry->staged = std::move(parts);
        } catch (const std::exception &) {} // load() reports the failure.
        changed_.notify_all();
    }
}

// Writes the entry's parts to its file without the lock.
bool CacheStore::write(std::unique_lock<std::mutex> & lock, Entry & entry) {
    entry.writing = true;
    ++busy_;
    disk_used_ += entry.file_bytes;
    const auto path = path_of(entry.meta.key);
    const StoreMeta meta = entry.meta;
    const auto parts = entry.ram;
    lock.unlock();
    std::string error;
    try {
        std::error_code ignored;
        fs::create_directories(fs::path(path).parent_path(), ignored);
        write_file(path, meta, parts, config_.fingerprint);
    } catch (const std::exception & e) { error = e.what(); }
    lock.lock();
    entry.writing = false;
    --busy_;
    changed_.notify_all();
    if (!error.empty()) {
        // The disk tier keeps what it has and takes nothing more.
        disk_used_ -= entry.file_bytes;
        disk_ok_ = false;
        totals_.error = error;
        return false;
    }
    entry.disk = true;
    totals_.written_bytes += entry.file_bytes;
    if (!entry.ram.empty()) shared_.insert(&entry);
    return true;
}

// One unit of background work; false when nothing is left to do.
bool CacheStore::step(std::unique_lock<std::mutex> & lock) {
    const bool disk = config_.disk_bytes && disk_ok_;
    const auto room = [&](const Entry & entry) { return disk_used_ + entry.file_bytes <= config_.disk_bytes; };
    // Frees host memory with one action: drops a copy that is also on disk,
    // writes the coldest blob out, or deletes the coldest leaf older than `stamp`.
    const auto free_ram = [&](uint64_t stamp) {
        for (auto * entry : ram_order_) {
            if (entry->disk) { drop_ram(*entry); return true; }
            if (entry->superseded) {
                if (!deletable(*entry)) continue;
                erase(*entry);
                return true;
            }
            if (disk) {
                if (room(*entry)) { if (write(lock, *entry)) return true; }
                else if (!shared_.empty()) {
                    // A blob that is also in memory gives up its file: nothing is lost.
                    Entry * hottest = nullptr;
                    for (auto * candidate : shared_)
                        if (!candidate->writing && !candidate->reading && (!hottest || Rank()(hottest, candidate))) hottest = candidate;
                    if (hottest) { drop_disk(*hottest); return true; }
                }
            }
            break;
        }
        auto * victim = coldest_leaf();
        if (!victim || !(victim->superseded || victim->stamp < stamp)) return false;
        erase(*victim);
        return true;
    };
    if (!waiters_.empty()) {
        auto & waiter = *waiters_.front();
        const bool fits = waiter.bytes > config_.ram_bytes ? oversize_ == 0 : ram_used_ + waiter.bytes <= config_.ram_bytes;
        if (!fits && free_ram(waiter.stamp)) return true;
        waiter.admitted = fits;
        waiter.done = true;
        waiters_.pop_front();
        changed_.notify_all();
        return true;
    }
    if (ram_used_ > config_.ram_bytes && !ram_order_.empty()) {
        // An oversize blob passing through: it is the first in line.
        if (free_ram((*ram_order_.begin())->stamp + 1)) return true;
    }
    if (disk && flushing_) {
        // Everything that deserves a file gets one, most recent first; a blob
        // on disk that is colder than the one waiting gives way to it.
        for (auto it = ram_order_.rbegin(); it != ram_order_.rend(); ++it) {
            Entry * entry = *it;
            if (entry->disk || entry->superseded) continue;
            if (room(*entry)) { write(lock, *entry); return true; }
            for (auto * leaf : leaf_order_) {
                if (!Rank()(leaf, entry)) break;
                if (!leaf->disk || !deletable(*leaf)) continue;
                erase(*leaf);
                return true;
            }
        }
    } else if (const uint64_t window = headroom(); disk && ram_used_ + window > config_.ram_bytes) {
        // Write ahead of need so that the coldest part of memory can be dropped at once.
        uint64_t seen = 0;
        for (auto * entry : ram_order_) {
            if (seen >= window) break;
            seen += entry->bytes;
            if (entry->disk || entry->superseded) continue;
            if (!room(*entry)) break;
            write(lock, *entry);
            return true;
        }
    }
    if (flushing_) {
        flushing_ = false;
        changed_.notify_all();
    }
    return false;
}

void CacheStore::spill() {
    std::unique_lock lock(mutex_);
    for (;;) {
        work_.wait(lock, [&] { return stop_ || pending_; });
        if (stop_) return;
        pending_ = false;
        while (!stop_ && step(lock)) {}
        changed_.notify_all();
    }
}

void CacheStore::flush() {
    std::unique_lock lock(mutex_);
    if (!(config_.disk_bytes && disk_ok_) || stop_) return;
    flushing_ = true;
    wake();
    changed_.wait(lock, [&] { return !flushing_ || stop_; });
}

void CacheStore::clear() {
    std::unique_lock lock(mutex_);
    read_queue_.clear();
    changed_.wait(lock, [&] { return busy_ == 0; });
    for (auto & [key, entry] : index_)
        if (entry->disk) {
            std::error_code ignored;
            fs::remove(path_of(key), ignored);
        }
    ram_order_.clear(); leaf_order_.clear(); snapshots_.clear(); shared_.clear();
    index_.clear();
    ram_used_ = disk_used_ = oversize_ = 0;
    totals_.chunks = totals_.snapshots = totals_.tokens = 0;
    wake();
}

CacheStore::Info CacheStore::info() const {
    std::lock_guard lock(mutex_);
    Info info = totals_;
    info.ram_bytes = ram_used_;
    info.disk_bytes = disk_used_;
    return info;
}

} // namespace tidal
