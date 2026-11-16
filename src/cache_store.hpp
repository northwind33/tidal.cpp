#pragma once
// Two-tier store behind the prompt cache: sequence state that the engine no
// longer holds, kept in host memory and spilled to disk.
//
// State is cut into immutable blobs addressed by the hash of the token prefix
// they complete, so sequences that share a prefix share its blobs:
//   - a chunk holds the KV rows (and index/MTP rows) of a fixed token range;
//   - a snapshot holds the recurrent state at one position plus the KV rows
//     since the last chunk boundary.
// A snapshot can be restored when every chunk below it is present, so blobs
// form a tree: a chunk's parent is the chunk before it, a snapshot's parent
// the chunk that ends where its own rows start.
//
// New blobs enter host memory. A background thread writes the least recently
// used ones to disk when memory fills up and drops what fits in neither tier,
// leaves first, so that a blob never outlives the chunks it needs. Blobs read
// back from disk stay in memory while there is room.
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace tidal {

// An uninitialized byte buffer; blobs are shared between the store, the
// engine's live snapshots and in-flight disk writes.
struct Bytes {
    std::unique_ptr<uint8_t[]> data;
    size_t size = 0;
    explicit Bytes(size_t n) : data(new uint8_t[n]), size(n) {}
};
using Blob = std::shared_ptr<const Bytes>;

uint64_t hash_bytes(const void * data, size_t size, uint64_t seed = 0);
// prefix[i] identifies the first i positions: token ids and, at image
// positions, the image's content hash (`images` may be empty).
std::vector<uint64_t> prefix_chain(const std::vector<int32_t> & tokens, const std::vector<uint64_t> & images, size_t count);

enum class StoreKind : uint8_t { Chunk = 0, Snapshot = 1 };
// End: the end of a prompt. Generated: the end of a reply, taken when another
// sequence replaces it. Break: a position inside a prompt (a cache breakpoint
// or the point where two sequences part).
enum class SnapshotRole : uint8_t { End = 0, Generated = 1, Break = 2 };

struct StoreKey {
    StoreKind kind = StoreKind::Chunk;
    int32_t position = 0; // Length of the prefix the blob completes.
    uint64_t hash = 0;    // prefix_chain value at `position`.
    bool operator==(const StoreKey & other) const {
        return kind == other.kind && position == other.position && hash == other.hash;
    }
};
struct StoreKeyHash {
    size_t operator()(const StoreKey & key) const {
        return size_t(key.hash ^ (uint64_t(uint32_t(key.position)) * 0x9E3779B97F4A7C15ull) ^ uint64_t(key.kind));
    }
};

struct StoreMeta {
    StoreKey key;
    int32_t first = 0;             // The blob holds rows [first, key.position).
    uint64_t parent = 0;           // prefix_chain value at `first`; the parent chunk ends there.
    std::vector<int32_t> tokens;   // Tokens of [first, key.position).
    std::vector<uint64_t> images;  // Their image hashes; empty when all are text.
    // Snapshots only.
    SnapshotRole role = SnapshotRole::End;
    int32_t anchor_position = 0;   // End of the prompt whose request took the snapshot.
    uint64_t anchor_hash = 0;
    int32_t mtp_next = 0;
    uint64_t captured = 0;         // Captured logits at the end of the first part.
    std::vector<uint64_t> parts;   // Payload part sizes; set by put().
};

// What a prompt asks of the store.
struct StoreQuery {
    const std::vector<uint64_t> & prefix;  // prefix_chain of the prompt (size + 1 values).
    const std::vector<int32_t> & tokens;
    const std::vector<uint64_t> & images;  // Image hash per position.
    int live = 0;   // Rows [0, live) of the prompt are resident in the engine.
    int floor = 0;  // Only snapshots beyond this position are of use.
};
// A restorable snapshot and the chunks to load for it, ascending: those that
// reach beyond the query's live rows. The entries stay pinned until release().
struct StoreMatch {
    StoreMeta snapshot;
    std::vector<StoreMeta> chunks;
};

class CacheStore {
public:
    struct Config {
        uint64_t ram_bytes = 0, disk_bytes = 0;
        std::string directory;     // Required when disk_bytes > 0.
        uint64_t fingerprint = 0;  // Blobs of another model or layout are never mixed in.
    };
    struct Info {
        uint64_t ram_bytes = 0, disk_bytes = 0;
        uint64_t chunks = 0, snapshots = 0, tokens = 0;
        uint64_t written_bytes = 0, read_bytes = 0, dropped_bytes = 0;
        uint64_t stored = 0, dropped = 0, loaded = 0;
        std::string error; // Latest disk failure; the disk tier stops growing after one.
    };
    explicit CacheStore(const Config & config);
    ~CacheStore(); // Writes what memory holds to disk when there is room, then stops.
    CacheStore(const CacheStore &) = delete;
    CacheStore & operator=(const CacheStore &) = delete;

    // A new logical time; blobs used by one operation share its stamp.
    uint64_t tick();
    bool contains(const StoreKey & key) const;
    // Stores a blob whose parent chunk is present. Waits while host memory is
    // being spilled; returns false when there is no room that could be made
    // by dropping blobs older than `stamp`. An End snapshot supersedes earlier
    // End and Generated snapshots whose anchor lies on `path`, its sequence's
    // prefix_chain.
    bool put(StoreMeta meta, std::vector<Blob> parts, uint64_t stamp, const std::vector<uint64_t> * path = nullptr);
    // The deepest snapshot on the prompt's path that is complete, if any.
    std::optional<StoreMatch> find(const StoreQuery & query, uint64_t stamp);
    void release(const StoreMatch & match);
    // The engine resumed from this snapshot out of its own memory.
    void used(const StoreKey & snapshot, uint64_t stamp);
    // Payload parts, from memory or disk. Throws when the blob is gone or damaged.
    std::vector<Blob> load(const StoreKey & key);
    // Starts reading a blob that is only on disk.
    void prefetch(const StoreKey & key);
    // Blocks until every blob that fits on disk is written there.
    void flush();
    void clear();
    Info info() const;

private:
    struct Entry;
    struct Rank {
        bool operator()(const Entry * a, const Entry * b) const;
    };
    struct RamRank {
        bool operator()(const Entry * a, const Entry * b) const;
    };
    struct Waiter { uint64_t bytes = 0, stamp = 0; bool done = false, admitted = false; };

    Config config_;
    std::string root_;      // Directory of this fingerprint's files.
    uint64_t ram_soft_ = 0; // Above this, cold blobs are written ahead of need.
    mutable std::mutex mutex_;
    std::condition_variable work_, changed_;
    std::unordered_map<StoreKey, std::unique_ptr<Entry>, StoreKeyHash> index_;
    std::set<Entry *, RamRank> ram_order_; // Entries resident in host memory.
    std::set<Entry *, Rank> leaf_order_;   // Entries nothing depends on.
    std::set<Entry *> snapshots_, shared_; // shared_: in memory and on disk.
    std::deque<Waiter *> waiters_;
    std::deque<StoreKey> read_queue_;
    uint64_t clock_ = 0, ram_used_ = 0, disk_used_ = 0, oversize_ = 0;
    bool stop_ = false, pending_ = false, flushing_ = false, disk_ok_ = true;
    int busy_ = 0; // Disk reads and writes in flight.
    int lock_ = -1; // Holds the directory against a second process.
    Info totals_;
    std::thread spiller_;
    std::vector<std::thread> readers_;

    Entry * get(const StoreKey & key) const;
    Entry * parent_of(const Entry & entry) const;
    void link(Entry & entry);
    void unlink(Entry & entry);
    void touch_chain(Entry & entry, uint64_t stamp);
    void set_ram(Entry & entry, std::vector<Blob> parts);
    void drop_ram(Entry & entry);
    void drop_disk(Entry & entry);
    bool deletable(const Entry & entry) const;
    void erase(Entry & entry);
    void supersede(Entry & entry);
    bool promote(Entry & entry, const std::vector<Blob> & parts);
    Entry * coldest_leaf() const;
    std::string path_of(const StoreKey & key) const;
    void scan();
    void spill();
    bool step(std::unique_lock<std::mutex> & lock);
    bool write(std::unique_lock<std::mutex> & lock, Entry & entry);
    void read_loop();
    std::vector<Blob> read(std::unique_lock<std::mutex> & lock, Entry & entry);
    void wake() { pending_ = true; work_.notify_all(); }
};

} // namespace tidal
