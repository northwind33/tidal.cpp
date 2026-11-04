#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tidal {

// A deliberately small Dense derivative, not the released MoE checkpoint schema.
struct Config {
    int hidden = 128;
    int ffn = 256;
    int vocab = 256;
    int layers = 4;
    int context = 256;
    int branches = 4;
    int rank = 32;
    int q_heads = 4;
    int kv_heads = 2;
    int head_dim = 32;
    int rope_dim = 16;
    int gdn_key_heads = 2;
    int gdn_value_heads = 4;
    int gdn_dim = 32;
    int conv_kernel = 4;
    int index_heads = 4;
    int index_dim = 32;
    int block_size = 4;
    int top_blocks = 2;
    float eps = 1e-6f;
    float rope_base = 10000.0f;
    void validate() const;
};

struct Options {
    std::string backend = "cuda";
    int device = 0;
    int threads = 4;
};

struct Selection {
    int layer;
    std::vector<int32_t> tokens;
};

struct Step {
    int position;
    std::vector<float> logits;
    std::vector<Selection> selections;
};

struct Statistics {
    uint64_t device_graph_nodes = 0;
    uint64_t sparse_attention_calls = 0;
    uint64_t device_to_host_bytes = 0;
    uint64_t host_to_device_bytes = 0;
    double graph_ms = 0;
    double attention_ms = 0;
    double transfer_ms = 0;
};

class Engine {
public:
    explicit Engine(const std::string & path, Options options = {});
    ~Engine();
    Engine(const Engine &) = delete;
    Engine & operator=(const Engine &) = delete;
    Step decode(int32_t token);
    // Correctness-first prompt ingestion: exactly the same transitions as decode.
    std::vector<Step> prefill(const std::vector<int32_t> & tokens);
    void reset();
    const Config & config() const;
    const Statistics & statistics() const;
    std::string backend_name() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tidal
