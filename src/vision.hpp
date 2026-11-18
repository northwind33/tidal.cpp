#pragma once
// Qwen3-VL vision tower (the ViT and patch merger of Qwen3.5, Qwen3.8 and
// Qwen3.8-Flash-Next) read from an mmproj GGUF, with Qwen-VL preprocessing.
#include "internal.hpp"

#include <array>
#include <string>
#include <vector>

namespace tidal {

struct VisionConfig {
    int image_size = 0, patch = 0, hidden = 0, ffn = 0, layers = 0, heads = 0, output = 0, merge = 2;
    int position_side = 0; // Learned position table is position_side x position_side.
    float eps = 1e-6f;
    float mean[3] = {0.5f, 0.5f, 0.5f}, std[3] = {0.5f, 0.5f, 0.5f};
    int min_tokens = 8, max_tokens = 4096;
    int factor() const { return patch * merge; }
};

// A resized, normalized image ready for patch embedding.
struct VisionInput {
    int width = 0, height = 0;          // Pixels after resizing (multiples of patch * merge).
    std::vector<uint8_t> resized;       // RGB8 after resizing.
    std::vector<float> patches;         // [3 * patch * patch, patches] in merge order.
    std::vector<float> positions;       // [hidden, patches] interpolated learned positions.
    std::vector<int32_t> rope;          // [4 * patches]: rows, columns, rows, columns.
};

// Qwen-VL smart resize: both sides become multiples of `factor` close to the
// original aspect ratio, with the pixel count inside [min_pixels, max_pixels].
std::array<int, 2> smart_resize(int width, int height, int factor, int64_t min_pixels, int64_t max_pixels);
// Pillow-compatible bicubic resampling (a = -0.5, antialiased when shrinking,
// 22-bit fixed-point weights, horizontal then vertical pass).
std::vector<uint8_t> resize_bicubic(const std::vector<uint8_t> & rgb, int width, int height, int out_width, int out_height);

class VisionEncoder {
public:
    VisionEncoder(const std::string & path, const Options & options, int text_hidden);
    ~VisionEncoder();
    VisionEncoder(const VisionEncoder &) = delete;
    VisionEncoder & operator=(const VisionEncoder &) = delete;
    const VisionConfig & config() const { return config_; }
    // Pixel size after resizing.
    std::array<int, 2> resized(int width, int height) const;
    int tokens(int width, int height) const;
    VisionInput prepare(const Image & image) const;
    ImageEmbedding encode(const Image & image, Statistics & stats, Profiler & profiler);
    VisionInfo info() const;
    void placements(std::vector<TensorPlacement> & out) const;
private:
    VisionConfig config_;
    Options options_;
    Backend backend_{nullptr, ggml_backend_free};
    Context weights_ctx_{nullptr, ggml_free};
    Buffer weights_{nullptr, ggml_backend_buffer_free};
    std::map<std::string, ggml_tensor *> tensors_;
    std::vector<std::string> order_;                 // Tensor names in file order.
    std::vector<float> position_table_;              // [hidden, side * side], host copy.
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator_{nullptr, ggml_gallocr_free};
    bool cpu_ = false, flash_ = true;
    uint64_t parameters_ = 0, stored_bytes_ = 0;
    ggml_tensor * at(const std::string & name) const;
};

} // namespace tidal
