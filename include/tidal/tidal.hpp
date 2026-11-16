#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
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
    // Interleaved MRoPE pairs per (temporal, height, width) axis; they sum to
    // rope_dim / 2. Text positions are equal on every axis (plain NeoX RoPE);
    // image tokens rotate by their grid row and column.
    int rope_sections[3] = {0, 0, 0};
    // Placeholder token of image positions (<|image_pad|>), -1 when unknown.
    // PLE hashes it at those positions, as the released checkpoints do.
    int image_token = -1;
    bool is_recurrent(int layer) const {
        return recurrent_layers.empty() ? (layer + 1) % 4 != 0 : recurrent_layers.at(layer);
    }
    void validate() const;
};

enum class ProfileMode { Off, Phases, Nodes };

enum class SamplingBackend { Device, CPU };
enum class PrefillPrecision { FP16, BF16 };
// 16-bit type of the CPU-resident KV cache (default CPU KV placement).
enum class CpuKVPrecision { BF16, FP16 };
enum class KVPlacement { CPU, GPU };
// Where the vision tower's weights live; the encoder runs on that device.
// Auto follows Options::backend.
enum class VisionPlacement { Auto, GPU, CPU };

struct SamplingConfig {
    SamplingBackend backend = SamplingBackend::Device;
    float temperature = 0; // Zero preserves greedy generation.
    int top_k = 0;         // Zero or -1 disables top-k.
    float top_p = 1;
    float min_p = 0;
    float repeat_penalty = 1;
    float frequency_penalty = 0;
    float presence_penalty = 0;
    int penalty_last_n = 64; // -1: entire consumed sequence; 0: no penalties.
    uint64_t seed = 0;
    void validate() const;
    bool penalties() const {
        return penalty_last_n != 0 && (repeat_penalty != 1 || frequency_penalty != 0 || presence_penalty != 0);
    }
};

struct ProfileEvent {
    std::string category, name, phase, backend, op, dtype;
    std::vector<int64_t> shape;
    std::vector<std::string> input_types;
    std::vector<std::vector<int64_t>> input_shapes;
    int position = -1, layer = -1, head = -1, node = -1;
    uint64_t bytes = 0;
    double start_us = 0, wall_us = 0;
};

// Session cache of Engine::prefill_cached: KV rows and state snapshots of
// sequences the engine no longer holds, kept in host memory up to `ram_bytes`
// and spilled to files under `directory` up to `disk_bytes`. A prompt that
// returns to one of those sequences resumes from the stored state instead of
// being prefilled again. Files are tied to the model file and the options that
// shape the state, and a later engine with the same ones adopts them.
struct CacheStoreOptions {
    uint64_t ram_bytes = 0;
    uint64_t disk_bytes = 0;  // Requires `directory`.
    std::string directory;
    // KV rows per stored chunk; sequences share the chunks of a common prefix.
    // Zero selects 256, rounded up to a multiple of the QSA block size.
    int chunk_tokens = 0;
    // A prompt that parts from another sequence at least this many tokens past
    // the state it resumes from leaves a snapshot at the parting point, where
    // later prompts with the same prefix resume. Zero disables it.
    int branch_tokens = 512;
    bool enabled() const { return ram_bytes || disk_bytes; }
};

struct Options {
    std::string backend = "cuda";
    int device = 0;
    int threads = 4;
    ProfileMode profile = ProfileMode::Off;
    bool nvtx = false;
    uint64_t profile_event_limit = 500000;
    int context = 0; // 0 uses GGUF capacity; otherwise bounded by the model capacity.
    SamplingConfig sampling;
    PrefillPrecision prefill_precision = PrefillPrecision::FP16;
    // Zero selects a memory-aware size. Explicit sizes are upper bounds; the
    // final range may be shorter. Neither range traverses the next layer.
    int prefill_chunk_tokens = 0;
    int prefill_attention_tile = 0;
    // Zero uses live device availability. A nonzero cap limits resident weights
    // plus prefill workspace, for controlled memory-pressure experiments.
    uint64_t prefill_memory_limit = 0;
    // UINT64_MAX selects automatic kernel/runtime headroom; zero disables it.
    uint64_t prefill_memory_reserve = UINT64_MAX;
    // Benchmark alternative: persistent CUDA KV and attention in both phases.
    // GPU KV uses prefill_precision directly; CPU KV remains canonical BF16.
    KVPlacement kv_placement = KVPlacement::CPU;
    // BF16 keeps the F32 exponent range. FP16 has a 10-bit mantissa and matches
    // the FP16 prefill/attention format, but rejects |K| or |V| above 65504 with
    // an error (no silent saturation or fallback).
    CpuKVPrecision cpu_kv_precision = CpuKVPrecision::BF16;
    // Prompt cache (prefill_cached): KV rows can be truncated, but GDN, PLE
    // convolution and QSA ring state cannot be rewound, so a rollback restores
    // the latest host-memory snapshot taken at the end of a prefill_cached call.
    // The oldest snapshot is dropped beyond this count; zero keeps only reuse of
    // a sequence that the new prompt extends.
    int prompt_cache_checkpoints = 4;
    // CUDA with CPU KV: prefills of at most this many tokens run as hybrid
    // batches (GPU sublayers over all tokens at once with resident decode
    // weights, attention on the CPU KV) instead of layer-major GPU prefill,
    // which uploads the whole KV prefix per attention layer. Zero disables.
    int hybrid_batch_tokens = 96;
    // Speculative decoding (CUDA with CPU KV): Engine::verify checks up to this
    // many drafted tokens per call. Zero disables it and its state snapshots.
    int max_draft_tokens = 0;
    // Load the model's MTP block for Engine::draft (Qwen3.5 architecture). A
    // model file without one needs mtp_model, a sidecar GGUF of the MTP tensors.
    bool mtp = false;
    std::string mtp_model;
    // Vision tower: an mmproj GGUF of the Qwen3-VL ViT and merger
    // (clip.projector_type qwen3vl_merger), as published for Qwen3.5, Qwen3.8
    // and Qwen3.8-Flash-Next (qwen4exp). Empty disables image input.
    std::string vision_model;
    VisionPlacement vision_placement = VisionPlacement::Auto;
    // Image token budget after resizing (one token per 2x2 merged patches);
    // zero keeps the defaults of 8 and 4096 tokens.
    int image_min_tokens = 0, image_max_tokens = 0;
    CacheStoreOptions cache_store;
};

// Decoded 8-bit RGB pixels, row-major, width * height * 3 bytes.
struct Image {
    int width = 0, height = 0;
    std::vector<uint8_t> rgb;
    // Content hash of the size and pixels; keys prompt-cache reuse.
    uint64_t hash() const;
};
// PNG (any bit depth, palette, alpha, interlace), baseline/progressive JPEG,
// BMP and binary PPM/PGM. Alpha is composited over white. Throws
// std::invalid_argument for unsupported or malformed data.
Image decode_image(const std::string & bytes);

// Output of the vision tower for one image: grid_height * grid_width merged
// tokens in row-major order, each of Config::hidden values.
struct ImageEmbedding {
    int grid_width = 0, grid_height = 0;
    int pixel_width = 0, pixel_height = 0; // After resizing.
    std::vector<float> values;
    uint64_t hash = 0;
    int tokens() const { return grid_width * grid_height; }
};
// One image of a prompt: its placeholder tokens are [offset, offset + count)
// of the token vector passed alongside, count = Engine::image_tokens(image).
// `embedding`, when set, is used instead of encoding `image` (which may then
// be null); a span reused by the prompt cache needs neither to be evaluated.
struct ImageSpan {
    int offset = 0;
    std::shared_ptr<const Image> image;
    std::shared_ptr<const ImageEmbedding> embedding;
};

struct VisionInfo {
    bool enabled = false;
    std::string placement;  // "gpu" or "cpu"
    std::string device;     // ggml backend executing the encoder
    int layers = 0, hidden = 0, heads = 0, ffn = 0, patch_size = 0, merge = 0, output = 0;
    int min_tokens = 0, max_tokens = 0;
    uint64_t parameters = 0, weight_bytes = 0;
};

// One tensor of the loaded model and the unit that computes with it.
struct TensorPlacement {
    std::string name;
    std::string component;  // text | mtp | vision
    std::string role;       // weight | state | kv_cache
    std::string group;      // Weight group or layer the tensor belongs to.
    std::string type;       // ggml type as stored in memory.
    std::vector<int64_t> shape;
    uint64_t bytes = 0;
    bool resident = false;  // Allocated now (prefill may evict weight groups).
    std::string memory;     // Buffer holding it: e.g. CUDA0, CUDA_Host, CPU.
    std::string compute;    // Device whose kernels read it: e.g. CUDA0 or CPU.
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
    uint64_t sampling_calls = 0;
    uint64_t sampling_device_to_host_bytes = 0;
    uint64_t sampling_host_to_device_bytes = 0;
    double sampling_ms = 0; // Inclusive of sampler transfers; overlaps transfer_ms.
    uint64_t prefill_tokens = 0, prefill_layers = 0, prefill_gpu_attention_calls = 0;
    uint64_t prefill_kv_to_host_bytes = 0;
    uint64_t prefill_weight_evictions = 0, prefill_weight_reloads = 0, prefill_weight_prefetches = 0;
    uint64_t prefill_weight_reload_bytes = 0, prefill_peak_workspace_bytes = 0;
    uint64_t prefill_chunks = 0, prefill_chunk_retries = 0;
    uint64_t prefill_chunk_min_tokens = 0, prefill_chunk_max_tokens = 0;
    uint64_t prefill_attention_min_tile = 0, prefill_attention_max_tile = 0;
    uint64_t prefill_reserve_bytes = 0;
    uint64_t gpu_attention_calls = 0; // Decode only; prefill has its own counter.
    uint64_t decode_graph_reuses = 0; // GPU KV decode tokens that reused the previous graph.
    uint64_t gpu_kv_bytes = 0; // Persistent, padded K/V allocation; survives reset.
    uint64_t prompt_cache_reused_tokens = 0; // prefill_cached positions not re-evaluated.
    uint64_t prompt_cache_restores = 0;      // Rollbacks to a snapshot (or to an empty sequence).
    uint64_t prompt_cache_checkpoint_bytes = 0; // Host memory held by current snapshots.
    uint64_t prompt_cache_pinned = 0;           // Current snapshots exempt from the count limit.
    uint64_t cache_store_restores = 0;          // prefill_cached calls that resumed from the cache store.
    uint64_t cache_store_restored_tokens = 0;   // Positions they took from it beyond the engine's own reuse.
    uint64_t cache_store_saved_bytes = 0;       // KV rows and snapshots copied into the store.
    double cache_store_restore_ms = 0, cache_store_save_ms = 0;
    uint64_t speculative_verifications = 0, speculative_drafted = 0, speculative_accepted = 0;
    uint64_t mtp_drafts = 0; // Engine::draft calls.
    uint64_t images_encoded = 0, image_tokens = 0; // Vision tower runs and their output tokens.
    double vision_ms = 0;                          // Preprocessing and encoding.
};

struct CacheStoreInfo {
    bool enabled = false;
    uint64_t ram_bytes = 0, ram_limit = 0, disk_bytes = 0, disk_limit = 0;
    uint64_t chunks = 0, snapshots = 0, tokens = 0;  // Stored blobs; KV rows held by the chunks.
    uint64_t written_bytes = 0, read_bytes = 0;      // Disk traffic of this engine.
    uint64_t dropped = 0, dropped_bytes = 0;         // Blobs that no longer fit either budget.
    int chunk_tokens = 0;
    std::string error;                               // Latest disk failure; the disk tier stops growing after one.
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
    // Disable capture to keep the full vocabulary vector on the device.
    Step decode(int32_t token, bool capture_logits = true);
    // CUDA: all prompt tokens complete one layer before the next layer starts.
    // CPU: explicit sequential reference. Batched reductions can round differently.
    // Capture includes every position's logits and attention selections; disabling
    // it keeps vocabulary vectors and diagnostic selection lists off the host.
    std::vector<Step> prefill(const std::vector<int32_t> & tokens, bool capture_logits = true);
    // Makes `tokens` the whole sequence, reusing the longest consumed prefix the
    // cache can restore (the current sequence if `tokens` extends it, otherwise
    // the latest snapshot inside the common prefix) and prefilling the rest.
    // Returns steps for evaluated positions only; when nothing remains to be
    // evaluated, the result is empty and sample() uses the restored logits.
    // A snapshot of the final state is kept for later prompts.
    std::vector<Step> prefill_cached(const std::vector<int32_t> & tokens, bool capture_logits = true);
    // Prompts with images: each span's positions take the vision tower output
    // instead of token embeddings and rotate by MRoPE grid positions; text
    // after an image continues from the image's start plus max(rows, columns).
    // Placeholder tokens must equal Config::image_token when it is known.
    std::vector<Step> prefill(const std::vector<int32_t> & tokens, const std::vector<ImageSpan> & images,
                              bool capture_logits = true);
    // The cache matches images by content hash, and encodes only images that
    // reach past the reused prefix.
    std::vector<Step> prefill_cached(const std::vector<int32_t> & tokens, const std::vector<ImageSpan> & images,
                                     bool capture_logits = true);
    // As above, and additionally keeps a pinned snapshot at each of
    // `pin_positions` (prefix lengths in (0, tokens.size()]) that this call
    // evaluates or that already has a snapshot: the prefill stops at those
    // positions to take it. Positions inside the reused prefix without a
    // snapshot are skipped. Pinned snapshots are exempt from the
    // prompt_cache_checkpoints limit until unpinned by keep_checkpoints();
    // like all snapshots they are dropped when the sequence is rewound below
    // them or reset.
    std::vector<Step> prefill_cached(const std::vector<int32_t> & tokens, const std::vector<ImageSpan> & images,
                                     bool capture_logits, const std::vector<int> & pin_positions);
    // As above; `reuse` false evaluates the whole prompt. The sequence it
    // replaces is still saved to the cache store first.
    std::vector<Step> prefill_cached(const std::vector<int32_t> & tokens, const std::vector<ImageSpan> & images,
                                     bool capture_logits, const std::vector<int> & pin_positions, bool reuse);
    // Cache store (Options::cache_store). prefill_cached saves the state it is
    // about to overwrite and looks stored state up on its own; cache_sync does
    // the saving ahead of time, while the engine is idle: it copies the
    // snapshots of the current sequence and the KV rows below them that the
    // store lacks. `interrupt` is polled between blobs; the result is false
    // when it stopped the copy, which a later call continues.
    bool cache_sync(const std::function<bool()> & interrupt = {});
    // Saves the current sequence up to its last position and waits until the
    // store has written what fits on disk; for an orderly shutdown.
    void cache_flush();
    // Drops everything the store holds, files included.
    void cache_clear();
    CacheStoreInfo cache_store() const;
    // Pins the snapshots at `positions` and unpins all others; unpinned ones
    // beyond prompt_cache_checkpoints are dropped, oldest first.
    void keep_checkpoints(const std::vector<int> & positions);
    // Positions of the current snapshots, ascending, and whether each is pinned.
    std::vector<std::pair<int, bool>> checkpoints() const;
    // Vision (Options::vision_model).
    bool has_vision() const;
    VisionInfo vision_info() const;
    // Placeholder tokens of an image after resizing; a pure function of its size.
    int image_tokens(int width, int height) const;
    ImageEmbedding encode_image(const Image & image);
    // Every weight (text, MTP, vision), KV cache and recurrent state tensor
    // with its memory and compute device.
    std::vector<TensorPlacement> tensor_placements() const;
    // Every consumed token of the current sequence, in order.
    const std::vector<int32_t> & tokens() const;
    // Speculative decoding (Options::max_draft_tokens > 0). Consumes `token`
    // and verifies `drafts` after it in one batch: positions are sampled in
    // order with the decode sampler (penalties, RNG) and a draft is accepted
    // while it equals the sample. Returns the accepted drafts followed by the
    // first sample that is not an accepted draft; that last token is not
    // consumed and is the next call's `token`. With deterministic drafts the
    // output distribution equals decode + sample.
    std::vector<int32_t> verify(int32_t token, const std::vector<int32_t> & drafts);
    // MTP (Options::mtp): greedy guesses of up to `count` tokens that follow
    // `token`, the sampled but not yet consumed next token.
    std::vector<int32_t> draft(int32_t token, int count);
    // Draw from the last decoded position; consumes RNG but no context token.
    int32_t sample();
    // Replaces the sampling parameters and reseeds the generator, keeping the
    // sampler backend chosen at construction. Penalty history is rebuilt from
    // the consumed sequence, so it may be called between prompts or mid-sequence.
    void set_sampling(const SamplingConfig & sampling);
    // Vocabulary logits of the last consumed position (the distribution that
    // sample() draws from, before penalties); a device-to-host copy on CUDA.
    std::vector<float> logits();
    std::string sampling_backend_name() const;
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
