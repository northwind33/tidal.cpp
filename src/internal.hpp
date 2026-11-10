#pragma once

#include "tidal/tidal.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "gguf.h"
#include <map>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <stdexcept>
#include <utility>

namespace tidal {
constexpr const char * architecture = "tidal-qwen4exp-dense";
constexpr const char * moe_architecture = "tidal-qwen4exp-moe";
constexpr uint32_t format_version = 1;

using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using Gguf = std::unique_ptr<gguf_context, decltype(&gguf_free)>;
using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
using Backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;

struct TensorSpec {
    std::string name;
    int64_t columns;
    int64_t rows;
    enum Kind { Matrix, Norm, Conv, ALog, Bias, Embedding, Lookup, Router, Expert } kind;
    int64_t slices = 1;
    std::string source_name; // Empty means the engine name is also the GGUF name.
};
std::vector<TensorSpec> tensor_specs(const Config & c);
bool storage_type(ggml_type type);
std::vector<ggml_type> storage_types();
ggml_type parse_storage_type(const std::string & name);
void dequantize(ggml_type type, const void * data, float * values, int64_t count);
void write_config(gguf_context * file, const Config & config);
Config read_config(const gguf_context * file);
Config read_qwen35_config(const gguf_context * file);
void validate_qwen35(const Config & config);
std::vector<TensorSpec> qwen35_tensor_specs(const Config & config);
bool qwen35_mtp_tensor(const Config & config, const std::string & name);
void validate_ple(const Config & config);
void write_ple(gguf_context * file, const PLEConfig & config);
void read_ple(const gguf_context * file, Config & config);
// Previous tokens are oldest-first. Only the ngram_size-1 most recent are needed.
std::vector<int32_t> ple_rows(const PLEConfig & config, const PLELayer & layer,
                              int32_t token, const std::vector<int32_t> & previous);

enum class WeightPart { Embedding, Attention, FeedForward, PLE, Head };
struct WeightGroup {
    int layer = -1;
    WeightPart part = WeightPart::Embedding;
    std::string name;
    std::vector<ggml_tensor *> tensors;
    Buffer buffer{nullptr, ggml_backend_buffer_free};
    size_t allocation_bytes = 0, payload_bytes = 0;
    std::future<void> pending;
};

struct Model {
    Config config;
    ModelInfo info;
    Context ctx{nullptr, ggml_free};
    std::vector<WeightGroup> groups;
    std::map<std::string, ggml_tensor *> weights;
    Model(const std::string & path, ggml_backend_t backend);
    ~Model();
    Model(const Model &) = delete;
    Model & operator=(const Model &) = delete;
    ggml_tensor * at(const std::string & name) const { return weights.at(name); }
    int group(int layer, WeightPart part) const;
    bool resident(int id) const;
    size_t resident_bytes() const;
    void load(int id, bool asynchronous = false);
    void wait(int id);
    void wait_all();
    void evict(int id);
private:
    struct Backing;
    std::unique_ptr<Backing> backing;
};

class Profiler;
// Persistent workers for CPU attention. run() executes fn(0..tasks-1) on the
// caller and the workers and returns when every task has finished.
class WorkerPool {
public:
    explicit WorkerPool(int threads);
    ~WorkerPool();
    WorkerPool(const WorkerPool &) = delete;
    WorkerPool & operator=(const WorkerPool &) = delete;
    int threads() const { return int(workers_.size()) + 1; }
    void run(int tasks, const std::function<void(int)> & fn);
private:
    void work();
    void drain(uint64_t generation, int tasks, const std::function<void(int)> * job);
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable wake_;
    // Upper 32 bits: dispatch generation; lower 32 bits: next task index. A
    // worker claims a task only while the generation it observed is current, so
    // a late worker can never run a finished dispatch's job.
    std::atomic<uint64_t> ticket_{0};
    std::atomic<int> finished_{0};
    const std::function<void(int)> * job_ = nullptr;
    int tasks_ = 0;
    uint64_t generation_ = 0;
    bool stop_ = false;
};

class SparseAttention {
public:
    SparseAttention(int heads, int kv_heads, int dim, int capacity, int threads,
                    std::shared_ptr<WorkerPool> pool = nullptr, ggml_type type = GGML_TYPE_BF16);
    void append(const std::vector<float> & key, const std::vector<float> & value);
    // Appends rows already in the cache's 16-bit type(); FP16 rows holding
    // Inf/NaN (a device-side overflow) are rejected with std::overflow_error.
    void append_raw(const uint16_t * key, const uint16_t * value, int tokens);
    ggml_type type() const { return type_; }
    // Copies rows [first, first + count) out in token-major [token][kv_head][dim] order.
    void copy_rows(int first, int count, uint16_t * key, uint16_t * value) const;
    std::vector<float> attend(const std::vector<float> & query,
                              const std::vector<int32_t> & selected, Profiler * profiler = nullptr,
                              int position = -1, int layer = -1,
                              const std::string & phase = "qsa.attention") const;
    void reset() { size_ = 0; }
    // Drops rows [tokens, size()); prompt-cache rollback rewrites them later.
    void truncate(int tokens) {
        if (tokens < 0 || tokens > size_) throw std::out_of_range("KV truncation beyond the cached rows");
        size_ = tokens;
    }
    int size() const { return size_; }
private:
    void store(const uint16_t * key, const uint16_t * value, int tokens);
    size_t row_offset(int head, int token) const { return (size_t(head) * capacity_ + token) * dim_; }
    int heads_, kv_heads_, dim_, capacity_, threads_, size_ = 0;
    ggml_type type_;
    std::shared_ptr<WorkerPool> pool_;
    std::vector<uint16_t> keys_, values_;
};
std::vector<int32_t> expand_blocks(const std::vector<int32_t> & blocks,
                                  int visible, int block_size);
} // namespace tidal
