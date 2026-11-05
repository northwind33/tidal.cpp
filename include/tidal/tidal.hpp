#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tidal {

struct PLELayer {
    int layer = 0; // Zero-based decoder layer, in ascending order.
    int vocab = 0; // Includes padding rows in the embedding table.
    std::vector<uint64_t> multipliers, offsets, vocab_sizes;
};

struct PLEConfig {
    int ngram_size = 3;
    int heads_per_ngram = 2;
    int embedding_dim = 128; // Concatenated heads.
    int conv_kernel = 4;
    int eos_token_id = 0;
    std::vector<PLELayer> layers; // Empty preserves the original PLE-free schema.
    int heads() const { return (ngram_size - 1) * heads_per_ngram; }
};

enum class Architecture { Qwen4Exp, Qwen35 };

struct Config {
    Architecture architecture = Architecture::Qwen4Exp;
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
    int experts = 0;
    int experts_used = 0;
    int expert_ffn = 0;
    int shared_ffn = 0;
    bool norm_topk = true;
    PLEConfig ple;
    // Native qwen35 files may declare a recurrent-layer mask and an MTP tail.
    std::vector<bool> recurrent_layers;
    bool tied_embeddings = false;
    int mtp_layers = 0;
    int eos_token = -1;
    bool is_recurrent(int layer) const {
        return recurrent_layers.empty() ? (layer + 1) % 4 != 0 : recurrent_layers.at(layer);
    }
    void validate() const;
};

enum class ProfileMode { Off, Phases, Nodes };

struct ProfileEvent {
    std::string category, name, phase, backend, op, dtype;
    std::vector<int64_t> shape;
    std::vector<std::string> input_types;
    std::vector<std::vector<int64_t>> input_shapes;
    int position = -1, layer = -1, head = -1, node = -1;
    uint64_t bytes = 0;
    double start_us = 0, wall_us = 0;
};

struct Options {
    std::string backend = "cuda";
    int device = 0;
    int threads = 4;
    ProfileMode profile = ProfileMode::Off;
    bool nvtx = false;
    uint64_t profile_event_limit = 500000;
    int context = 0; // 0 uses GGUF capacity; otherwise bounded by the model capacity.
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
    uint64_t full_attention_calls = 0;
    uint64_t device_to_host_bytes = 0;
    uint64_t host_to_device_bytes = 0;
    double graph_ms = 0;
    double attention_ms = 0;
    double transfer_ms = 0;
};

struct ModelInfo {
    uint64_t parameters = 0;
    uint64_t stored_weight_bytes = 0;
    uint64_t resident_weight_bytes = 0;
    uint64_t weight_buffer_bytes = 0;
    uint64_t converted_tensors = 0;
    uint64_t skipped_tensors = 0; // Unused, explicitly declared MTP tensors.
    uint64_t skipped_weight_bytes = 0;
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
    const ModelInfo & model_info() const;
    std::string backend_name() const;
    std::string backend_description() const;
    // Changes instrumentation and clears events without resetting the sequence.
    // reset() also clears events. Read profile data only between decode calls.
    void set_profile(ProfileMode mode);
    const std::vector<ProfileEvent> & profile_events() const;
    uint64_t profile_dropped_events() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tidal
