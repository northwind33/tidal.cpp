#include "vision.hpp"
#include "profiler.hpp"
#include "ggml-cpu.h"
#ifdef TIDAL_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>

namespace tidal {
namespace {
using Clock = std::chrono::steady_clock;

const gguf_context * gguf_ptr(const Gguf & file) { return file.get(); }
int64_t find_key(const gguf_context * f, const char * name) { return gguf_find_key(f, name); }
uint32_t key_u32(const gguf_context * f, const char * name, int64_t fallback = -1) {
    const auto id = find_key(f, name);
    if (id < 0) {
        if (fallback >= 0) return uint32_t(fallback);
        throw std::runtime_error(std::string("Vision GGUF lacks ") + name);
    }
    if (gguf_get_kv_type(f, id) != GGUF_TYPE_UINT32 && gguf_get_kv_type(f, id) != GGUF_TYPE_INT32)
        throw std::runtime_error(std::string("Vision GGUF key has the wrong type: ") + name);
    return gguf_get_val_u32(f, id);
}
float key_f32(const gguf_context * f, const char * name, float fallback) {
    const auto id = find_key(f, name);
    if (id < 0) return fallback;
    if (gguf_get_kv_type(f, id) != GGUF_TYPE_FLOAT32) throw std::runtime_error(std::string("Vision GGUF key has the wrong type: ") + name);
    return gguf_get_val_f32(f, id);
}
std::string key_string(const gguf_context * f, const char * name) {
    const auto id = find_key(f, name);
    if (id < 0 || gguf_get_kv_type(f, id) != GGUF_TYPE_STRING) return {};
    return gguf_get_val_str(f, id);
}
void key_triple(const gguf_context * f, const char * name, float out[3]) {
    const auto id = find_key(f, name);
    if (id < 0) return;
    if (gguf_get_kv_type(f, id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(f, id) != GGUF_TYPE_FLOAT32 || gguf_get_arr_n(f, id) != 3)
        throw std::runtime_error(std::string("Vision GGUF key must hold three floats: ") + name);
    std::memcpy(out, gguf_get_arr_data(f, id), 3 * sizeof(float));
}

// Matrices keep their stored type when the backend multiplies it natively.
bool native_matrix(ggml_backend_t backend, ggml_tensor * stored, int64_t columns, bool cpu) {
    if (!storage_type(stored->type)) return false;
    if (cpu && stored->type != GGML_TYPE_F32 && !ggml_get_type_traits_cpu(stored->type)->vec_dot) return false;
    Context probe(ggml_init({65536, nullptr, true}), ggml_free);
    auto * w = ggml_new_tensor_2d(probe.get(), stored->type, columns, 64);
    auto * x = ggml_new_tensor_2d(probe.get(), GGML_TYPE_F32, columns, 64);
    auto * op = ggml_mul_mat(probe.get(), w, x);
    ggml_prec_set_acc(op, GGML_PREC_F32);
    return ggml_backend_supports_op(backend, op);
}

int round_even(double value) { return int(std::nearbyint(value)); }
} // namespace

std::array<int, 2> smart_resize(int width, int height, int factor, int64_t min_pixels, int64_t max_pixels) {
    if (width <= 0 || height <= 0) throw std::invalid_argument("Image has no pixels");
    if (double(std::max(width, height)) / std::min(width, height) > 200)
        throw std::invalid_argument("Image aspect ratio must be smaller than 200:1");
    // As the Qwen-VL processors: round to the factor (Python round(), half to
    // even), then scale down to max_pixels or up to min_pixels.
    const float f = float(factor);
    int w = std::max(factor, round_even(float(width) / f) * factor);
    int h = std::max(factor, round_even(float(height) / f) * factor);
    if (int64_t(w) * h > max_pixels) {
        const float beta = std::sqrt(float(height) * float(width) / float(max_pixels));
        h = std::max(factor, int(std::floor(float(height) / beta / f)) * factor);
        w = std::max(factor, int(std::floor(float(width) / beta / f)) * factor);
    } else if (int64_t(w) * h < min_pixels) {
        const float beta = std::sqrt(float(min_pixels) / (float(height) * float(width)));
        h = int(std::ceil(float(height) * beta / f)) * factor;
        w = int(std::ceil(float(width) * beta / f)) * factor;
    }
    return {w, h};
}

std::vector<uint8_t> resize_bicubic(const std::vector<uint8_t> & rgb, int width, int height, int out_width, int out_height) {
    if (rgb.size() != size_t(width) * height * 3 || out_width <= 0 || out_height <= 0) throw std::invalid_argument("Invalid resize");
    constexpr int bits = 22; // 32 - 8 (pixel) - 2 (accumulation headroom)
    struct Coefficients { int size = 0; std::vector<int> first, count; std::vector<int32_t> weights; };
    const auto filter = [](double x) {
        constexpr double a = -0.5;
        x = std::fabs(x);
        if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
        if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
        return 0.0;
    };
    const auto coefficients = [&](int in, int out) {
        Coefficients c;
        const double scale = double(in) / out, filter_scale = std::max(1.0, scale), support = 2.0 * filter_scale;
        c.size = int(std::ceil(support)) * 2 + 1;
        c.first.resize(size_t(out)); c.count.resize(size_t(out)); c.weights.assign(size_t(out) * c.size, 0);
        std::vector<double> k(size_t(c.size));
        for (int o = 0; o < out; ++o) {
            const double center = (o + 0.5) * scale;
            const int first = std::max(0, int(center - support + 0.5));
            const int count = std::min(in, int(center + support + 0.5)) - first;
            double sum = 0;
            for (int i = 0; i < count; ++i) { k[size_t(i)] = filter((i + first - center + 0.5) / filter_scale); sum += k[size_t(i)]; }
            for (int i = 0; i < count; ++i) {
                const double w = sum != 0 ? k[size_t(i)] / sum : k[size_t(i)];
                // Pillow rounds half away from zero by truncating w * 2^bits +- 0.5.
                c.weights[size_t(o) * c.size + i] = int32_t(w * double(1 << bits) + (w < 0 ? -0.5 : 0.5));
            }
            c.first[size_t(o)] = first; c.count[size_t(o)] = count;
        }
        return c;
    };
    const auto clip = [](int32_t v) { return uint8_t(v < 0 ? 0 : v > 255 ? 255 : v); };
    std::vector<uint8_t> current = rgb;
    int cw = width;
    if (out_width != width) {
        const auto c = coefficients(width, out_width);
        std::vector<uint8_t> next(size_t(out_width) * height * 3);
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < out_width; ++x) {
                const uint8_t * p = current.data() + (size_t(y) * cw + c.first[size_t(x)]) * 3;
                const int32_t * k = c.weights.data() + size_t(x) * c.size;
                int32_t s[3] = {1 << (bits - 1), 1 << (bits - 1), 1 << (bits - 1)};
                for (int i = 0; i < c.count[size_t(x)]; ++i, p += 3)
                    for (int ch = 0; ch < 3; ++ch) s[ch] += int32_t(p[ch]) * k[i];
                for (int ch = 0; ch < 3; ++ch) next[(size_t(y) * out_width + x) * 3 + ch] = clip(s[ch] >> bits);
            }
        current.swap(next); cw = out_width;
    }
    if (out_height != height) {
        const auto c = coefficients(height, out_height);
        std::vector<uint8_t> next(size_t(cw) * out_height * 3);
        for (int y = 0; y < out_height; ++y) {
            const int32_t * k = c.weights.data() + size_t(y) * c.size;
            for (int x = 0; x < cw; ++x) {
                int32_t s[3] = {1 << (bits - 1), 1 << (bits - 1), 1 << (bits - 1)};
                for (int i = 0; i < c.count[size_t(y)]; ++i) {
                    const uint8_t * p = current.data() + (size_t(c.first[size_t(y)] + i) * cw + x) * 3;
                    for (int ch = 0; ch < 3; ++ch) s[ch] += int32_t(p[ch]) * k[i];
                }
                for (int ch = 0; ch < 3; ++ch) next[(size_t(y) * cw + x) * 3 + ch] = clip(s[ch] >> bits);
            }
        }
        current.swap(next);
    }
    return current;
}

VisionEncoder::VisionEncoder(const std::string & path, const Options & options, int text_hidden) : options_(options) {
    ggml_context * raw = nullptr;
    Gguf file(gguf_init_from_file(path.c_str(), {true, &raw}), gguf_free);
    Context meta(raw, ggml_free);
    if (!file || !meta) throw std::runtime_error("Cannot read vision GGUF: " + path);
    const auto * f = gguf_ptr(file);
    if (key_string(f, "general.architecture") != "clip")
        throw std::runtime_error("Vision model must be an mmproj GGUF (general.architecture = clip)");
    auto projector = key_string(f, "clip.vision.projector_type");
    if (projector.empty()) projector = key_string(f, "clip.projector_type");
    if (projector != "qwen3vl_merger")
        throw std::runtime_error("Unsupported vision projector '" + projector + "': only the Qwen3-VL merger (qwen3vl_merger) is implemented");
    if (const auto id = find_key(f, "clip.has_vision_encoder"); id >= 0 && !gguf_get_val_bool(f, id))
        throw std::runtime_error("The mmproj GGUF has no vision encoder");
    auto & c = config_;
    c.image_size = int(key_u32(f, "clip.vision.image_size", 0));
    c.patch = int(key_u32(f, "clip.vision.patch_size"));
    c.hidden = int(key_u32(f, "clip.vision.embedding_length"));
    c.ffn = int(key_u32(f, "clip.vision.feed_forward_length"));
    c.layers = int(key_u32(f, "clip.vision.block_count"));
    c.heads = int(key_u32(f, "clip.vision.attention.head_count"));
    c.output = int(key_u32(f, "clip.vision.projection_dim", 0));
    c.merge = int(key_u32(f, "clip.vision.spatial_merge_size", 2));
    c.eps = key_f32(f, "clip.vision.attention.layer_norm_epsilon", 1e-6f);
    key_triple(f, "clip.vision.image_mean", c.mean);
    key_triple(f, "clip.vision.image_std", c.std);
    if (const auto id = find_key(f, "clip.use_gelu"); id >= 0 && !gguf_get_val_bool(f, id))
        throw std::runtime_error("The Qwen3-VL vision tower uses GELU; clip.use_gelu is false");
    if (const auto id = find_key(f, "clip.vision.is_deepstack_layers"); id >= 0) {
        const auto * flags = static_cast<const bool *>(gguf_get_arr_data(f, id));
        for (size_t i = 0; i < gguf_get_arr_n(f, id); ++i)
            if (flags[i]) throw std::runtime_error("DeepStack vision layers are not supported");
    }
    if (c.patch < 1 || c.patch > 64 || c.hidden < 8 || c.hidden > 8192 || c.ffn < 1 || c.ffn > 65536 || c.layers < 1 ||
        c.layers > 128 || c.heads < 1 || c.hidden % c.heads || (c.hidden / c.heads) % 4 || c.merge != 2 || !(c.eps > 0))
        throw std::runtime_error("Unsupported vision dimensions");
    // Token budget: llama.cpp's Qwen-VL defaults unless the file or options say otherwise.
    const int area = c.factor() * c.factor();
    if (const auto id = find_key(f, "clip.vision.image_min_pixels"); id >= 0) c.min_tokens = int(std::max<uint32_t>(1, gguf_get_val_u32(f, id) / area));
    if (const auto id = find_key(f, "clip.vision.image_max_pixels"); id >= 0) c.max_tokens = int(std::max<uint32_t>(1, gguf_get_val_u32(f, id) / area));
    if (options.image_min_tokens < 0 || options.image_max_tokens < 0) throw std::invalid_argument("Image token limits must be non-negative");
    if (options.image_min_tokens) c.min_tokens = options.image_min_tokens;
    if (options.image_max_tokens) c.max_tokens = options.image_max_tokens;
    if (c.min_tokens < 1 || c.max_tokens < c.min_tokens || c.max_tokens > 65536)
        throw std::invalid_argument("Image token limits must satisfy 1 <= min <= max <= 65536");

    auto * position = ggml_get_tensor(meta.get(), "v.position_embd.weight");
    if (!position || position->ne[0] != c.hidden) throw std::runtime_error("Vision GGUF lacks v.position_embd.weight");
    c.position_side = int(std::lround(std::sqrt(double(position->ne[1]))));
    if (int64_t(c.position_side) * c.position_side != position->ne[1]) throw std::runtime_error("Vision position table is not square");
    auto * merger_out = ggml_get_tensor(meta.get(), "mm.2.weight");
    if (!merger_out) throw std::runtime_error("Vision GGUF lacks mm.2.weight");
    if (c.output && c.output != merger_out->ne[1]) throw std::runtime_error("Vision projection_dim disagrees with mm.2.weight");
    c.output = int(merger_out->ne[1]);
    if (c.output != text_hidden)
        throw std::runtime_error("Vision tower output width " + std::to_string(c.output) + " does not match the text model's hidden size " +
                                 std::to_string(text_hidden) + "; the mmproj belongs to another model");

    // Placement.
    auto placement = options.vision_placement;
    if (placement == VisionPlacement::Auto) placement = options.backend == "cuda" ? VisionPlacement::GPU : VisionPlacement::CPU;
    if (placement != VisionPlacement::GPU && placement != VisionPlacement::CPU) throw std::invalid_argument("Invalid vision placement");
    cpu_ = placement == VisionPlacement::CPU;
    if (cpu_) {
        backend_.reset(ggml_backend_cpu_init());
        if (backend_) ggml_backend_cpu_set_n_threads(backend_.get(), options.threads);
    } else {
#ifdef TIDAL_CUDA
        if (options.device < 0 || options.device >= ggml_backend_cuda_get_device_count())
            throw std::runtime_error("Requested CUDA device is unavailable for the vision tower; no CPU fallback");
        backend_.reset(ggml_backend_cuda_init(options.device));
#else
        throw std::runtime_error("GPU vision placement requires a CUDA build; use --vision-placement cpu");
#endif
    }
    if (!backend_) throw std::runtime_error("Cannot initialize the vision backend");

    // Expected tensors and their resident shapes.
    struct Spec { std::string name; std::vector<int64_t> shape; bool matrix; };
    std::vector<Spec> specs;
    const int64_t pp = int64_t(c.patch) * c.patch * 3;
    specs.push_back({"v.patch_embd.weight", {pp, c.hidden}, true});
    specs.push_back({"v.patch_embd.weight.1", {pp, c.hidden}, true});
    specs.push_back({"v.patch_embd.bias", {c.hidden}, false});
    for (int l = 0; l < c.layers; ++l) {
        const auto p = "v.blk." + std::to_string(l) + ".";
        for (const char * n : {"ln1", "ln2"}) {
            specs.push_back({p + n + ".weight", {c.hidden}, false});
            specs.push_back({p + n + ".bias", {c.hidden}, false});
        }
        specs.push_back({p + "attn_qkv.weight", {c.hidden, 3 * int64_t(c.hidden)}, true});
        specs.push_back({p + "attn_qkv.bias", {3 * int64_t(c.hidden)}, false});
        specs.push_back({p + "attn_out.weight", {c.hidden, c.hidden}, true});
        specs.push_back({p + "attn_out.bias", {c.hidden}, false});
        specs.push_back({p + "ffn_up.weight", {c.hidden, c.ffn}, true});
        specs.push_back({p + "ffn_up.bias", {c.ffn}, false});
        specs.push_back({p + "ffn_down.weight", {c.ffn, c.hidden}, true});
        specs.push_back({p + "ffn_down.bias", {c.hidden}, false});
    }
    specs.push_back({"v.post_ln.weight", {c.hidden}, false});
    specs.push_back({"v.post_ln.bias", {c.hidden}, false});
    const int64_t merged = 4 * int64_t(c.hidden);
    specs.push_back({"mm.0.weight", {merged, merged}, true});
    specs.push_back({"mm.0.bias", {merged}, false});
    specs.push_back({"mm.2.weight", {merged, c.output}, true});
    specs.push_back({"mm.2.bias", {c.output}, false});

    weights_ctx_.reset(ggml_init({(specs.size() + 4) * ggml_tensor_overhead(), nullptr, true}));
    std::map<std::string, std::pair<ggml_tensor *, ggml_tensor *>> sources; // resident, stored
    for (const auto & spec : specs) {
        auto * stored = ggml_get_tensor(meta.get(), spec.name.c_str());
        if (!stored) throw std::runtime_error("Vision GGUF lacks " + spec.name);
        int64_t elements = 1;
        for (auto d : spec.shape) elements *= d;
        if (ggml_nelements(stored) != elements || stored->ne[0] != (spec.name.starts_with("v.patch_embd.weight") ? c.patch : spec.shape[0]))
            throw std::runtime_error("Unexpected vision tensor shape: " + spec.name);
        if (!storage_type(stored->type)) throw std::runtime_error("Unsupported vision tensor type: " + spec.name);
        const auto type = spec.matrix && native_matrix(backend_.get(), stored, spec.shape[0], cpu_) ? stored->type : GGML_TYPE_F32;
        auto * t = spec.shape.size() == 1 ? ggml_new_tensor_1d(weights_ctx_.get(), type, spec.shape[0])
                                          : ggml_new_tensor_2d(weights_ctx_.get(), type, spec.shape[0], spec.shape[1]);
        ggml_set_name(t, spec.name.c_str());
        tensors_[spec.name] = t;
        order_.push_back(spec.name);
        sources[spec.name] = {t, stored};
        parameters_ += uint64_t(elements);
        stored_bytes_ += ggml_nbytes(stored);
    }
    order_.push_back("v.position_embd.weight");
    parameters_ += uint64_t(ggml_nelements(position));
    stored_bytes_ += ggml_nbytes(position);
    for (int64_t id = 0; id < gguf_get_n_tensors(f); ++id) {
        const std::string name = gguf_get_tensor_name(f, id);
        if (!tensors_.contains(name) && name != "v.position_embd.weight")
            throw std::runtime_error("Unsupported vision tensor: " + name);
    }
    weights_.reset(ggml_backend_alloc_ctx_tensors(weights_ctx_.get(), backend_.get()));
    if (!weights_) throw std::runtime_error(std::string("Cannot allocate vision weights on ") + ggml_backend_name(backend_.get()));
    ggml_backend_buffer_set_usage(weights_.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open vision GGUF: " + path);
    const uint64_t base = gguf_get_data_offset(f);
    const auto read = [&](ggml_tensor * stored, std::vector<char> & bytes) {
        const auto id = gguf_find_tensor(f, ggml_get_name(stored));
        bytes.resize(ggml_nbytes(stored));
        input.seekg(std::streamoff(base + gguf_get_tensor_offset(f, id)));
        input.read(bytes.data(), std::streamsize(bytes.size()));
        if (!input) throw std::runtime_error(std::string("Truncated vision tensor: ") + ggml_get_name(stored));
    };
    std::vector<char> bytes;
    std::vector<float> values;
    for (const auto & [name, pair] : sources) {
        auto [t, stored] = pair;
        read(stored, bytes);
        if (t->type == stored->type) { ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size()); continue; }
        values.resize(size_t(ggml_nelements(stored)));
        dequantize(stored->type, bytes.data(), values.data(), ggml_nelements(stored));
        ggml_backend_tensor_set(t, values.data(), 0, values.size() * sizeof(float));
    }
    read(position, bytes);
    position_table_.resize(size_t(ggml_nelements(position)));
    dequantize(position->type, bytes.data(), position_table_.data(), ggml_nelements(position));
    std::cerr << "vision tower: " << c.layers << " layers, hidden " << c.hidden << ", output " << c.output << ", weights on "
              << ggml_backend_name(backend_.get()) << " (" << ggml_backend_buffer_get_size(weights_.get()) << " bytes)\n";
}

VisionEncoder::~VisionEncoder() = default;

ggml_tensor * VisionEncoder::at(const std::string & name) const { return tensors_.at(name); }

std::array<int, 2> VisionEncoder::resized(int width, int height) const {
    const int64_t area = int64_t(config_.factor()) * config_.factor();
    return smart_resize(width, height, config_.factor(), config_.min_tokens * area, config_.max_tokens * area);
}
int VisionEncoder::tokens(int width, int height) const {
    const auto size = resized(width, height);
    return (size[0] / config_.factor()) * (size[1] / config_.factor());
}

VisionInput VisionEncoder::prepare(const Image & image) const {
    if (image.width <= 0 || image.height <= 0 || image.rgb.size() != size_t(image.width) * image.height * 3)
        throw std::invalid_argument("Image must hold width * height RGB pixels");
    const auto & c = config_;
    VisionInput in;
    const auto size = resized(image.width, image.height);
    in.width = size[0]; in.height = size[1];
    in.resized = (in.width == image.width && in.height == image.height) ? image.rgb
        : resize_bicubic(image.rgb, image.width, image.height, in.width, in.height);
    const int pw = in.width / c.patch, ph = in.height / c.patch, n = pw * ph, k = c.patch * c.patch * 3;
    in.patches.resize(size_t(n) * k);
    in.positions.resize(size_t(n) * c.hidden);
    in.rope.resize(size_t(4) * n);
    // Learned positions, bilinear on the grid with aligned corners (as
    // torch.linspace(0, side - 1, size) indices in the reference).
    const int side = c.position_side;
    const auto axis = [&](int count) {
        std::vector<std::pair<int, float>> out(static_cast<size_t>(count));
        for (int i = 0; i < count; ++i) {
            const double pos = count > 1 ? double(i) * (side - 1) / (count - 1) : 0.0;
            const int lo = std::min(side - 1, int(std::floor(pos)));
            out[size_t(i)] = {lo, float(pos - lo)};
        }
        return out;
    };
    const auto rows = axis(ph), cols = axis(pw);
    float scale[3], shift[3];
    for (int ch = 0; ch < 3; ++ch) { scale[ch] = 1.0f / c.std[ch]; shift[ch] = c.mean[ch]; }
    int p = 0;
    for (int ty = 0; ty < ph / 2; ++ty)
        for (int tx = 0; tx < pw / 2; ++tx)
            for (int dy = 0; dy < 2; ++dy)
                for (int dx = 0; dx < 2; ++dx, ++p) {
                    const int py = 2 * ty + dy, px = 2 * tx + dx;
                    // Patch vector in the kernel's layout: column fastest, then row, then channel.
                    float * out = in.patches.data() + size_t(p) * k;
                    for (int ch = 0; ch < 3; ++ch)
                        for (int y = 0; y < c.patch; ++y)
                            for (int x = 0; x < c.patch; ++x) {
                                const uint8_t v = in.resized[((size_t(py) * c.patch + y) * in.width + size_t(px) * c.patch + x) * 3 + ch];
                                out[(ch * c.patch + y) * c.patch + x] = (float(v) / 255.0f - shift[ch]) * scale[ch];
                            }
                    const auto [y0, fy] = rows[size_t(py)];
                    const auto [x0, fx] = cols[size_t(px)];
                    const int y1 = std::min(side - 1, y0 + 1), x1 = std::min(side - 1, x0 + 1);
                    const float w00 = (1 - fy) * (1 - fx), w01 = (1 - fy) * fx, w10 = fy * (1 - fx), w11 = fy * fx;
                    const float * e00 = position_table_.data() + size_t(y0 * side + x0) * c.hidden;
                    const float * e01 = position_table_.data() + size_t(y0 * side + x1) * c.hidden;
                    const float * e10 = position_table_.data() + size_t(y1 * side + x0) * c.hidden;
                    const float * e11 = position_table_.data() + size_t(y1 * side + x1) * c.hidden;
                    float * dst = in.positions.data() + size_t(p) * c.hidden;
                    for (int i = 0; i < c.hidden; ++i) dst[i] = e00[i] * w00 + e01[i] * w01 + e10[i] * w10 + e11[i] * w11;
                    in.rope[size_t(p)] = py; in.rope[size_t(n + p)] = px;
                    in.rope[size_t(2 * n + p)] = py; in.rope[size_t(3 * n + p)] = px;
                }
    return in;
}

ImageEmbedding VisionEncoder::encode(const Image & image, Statistics & stats, Profiler & profiler) {
    const auto start = Clock::now();
    const auto & c = config_;
    ProfileScope scope(&profiler, "vision", "encode", "vision", -1, -1, ggml_backend_name(backend_.get()));
    ProfileScope prep(&profiler, "host", "image_preprocess", "vision", -1, -1, "CPU");
    const auto in = prepare(image);
    prep.finish();
    const int pw = in.width / c.patch, ph = in.height / c.patch, n = pw * ph, k = c.patch * c.patch * 3;
    const int d = c.hidden / c.heads;
    for (int attempt = 0; attempt < 2; ++attempt) {
        const size_t nodes = 64 * size_t(c.layers) + 256;
        Context ctx(ggml_init({nodes * ggml_tensor_overhead() * 2 + ggml_graph_overhead_custom(nodes, false), nullptr, true}), ggml_free);
        auto * g = ctx.get();
        auto * graph = ggml_new_graph_custom(g, nodes, false);
        auto * patches = ggml_new_tensor_2d(g, GGML_TYPE_F32, k, n); ggml_set_input(patches);
        auto * positions = ggml_new_tensor_2d(g, GGML_TYPE_F32, c.hidden, n); ggml_set_input(positions);
        auto * rope = ggml_new_tensor_1d(g, GGML_TYPE_I32, 4 * int64_t(n)); ggml_set_input(rope);
        const auto mm = [&](const std::string & w, ggml_tensor * x) {
            auto * out = ggml_mul_mat(g, at(w + ".weight"), x);
            ggml_prec_set_acc(out, GGML_PREC_F32);
            return ggml_add(g, out, at(w + ".bias"));
        };
        const auto layer_norm = [&](ggml_tensor * x, const std::string & w) {
            return ggml_add(g, ggml_mul(g, ggml_norm(g, x, c.eps), at(w + ".weight")), at(w + ".bias"));
        };
        // Qwen3-VL blocks use the tanh GELU; the CPU kernel approximates it with
        // an FP16 table, so the CPU path spells it out in F32.
        const auto gelu_tanh = [&](ggml_tensor * x) {
            if (!cpu_) return ggml_gelu(g, x);
            auto * cube = ggml_mul(g, ggml_sqr(g, x), x);
            auto * inner = ggml_scale(g, ggml_add(g, x, ggml_scale(g, cube, 0.044715f)), 0.7978845608028654f);
            return ggml_mul(g, x, ggml_scale_bias(g, ggml_tanh(g, inner), 0.5f, 0.5f));
        };
        // The temporal kernel halves see the same frame (a still image is
        // repeated over temporal_patch_size = 2).
        auto * x0 = ggml_mul_mat(g, at("v.patch_embd.weight"), patches);
        auto * x1 = ggml_mul_mat(g, at("v.patch_embd.weight.1"), patches);
        ggml_prec_set_acc(x0, GGML_PREC_F32); ggml_prec_set_acc(x1, GGML_PREC_F32);
        auto * x = ggml_add(g, ggml_add(g, ggml_add(g, x0, x1), at("v.patch_embd.bias")), positions);
        int sections[GGML_MROPE_SECTIONS] = {d / 4, d / 4, d / 4, d / 4};
        const float scale = 1.0f / std::sqrt(float(d));
        for (int l = 0; l < c.layers; ++l) {
            const auto p = "v.blk." + std::to_string(l) + ".";
            auto * h = layer_norm(x, p + "ln1");
            auto * qkv = mm(p + "attn_qkv", h);
            const auto view = [&](int part) {
                return ggml_view_3d(g, qkv, d, c.heads, n, size_t(d) * sizeof(float), qkv->nb[1], size_t(part) * c.hidden * sizeof(float));
            };
            // 2D rotary embedding: the first half of each head's rotated pairs
            // follows the patch row, the second half its column.
            auto * q = ggml_rope_multi(g, view(0), rope, nullptr, d / 2, sections, GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 0, 0);
            auto * kk = ggml_rope_multi(g, view(1), rope, nullptr, d / 2, sections, GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 0, 0);
            ggml_tensor * attended;
            if (flash_) {
                auto * qp = ggml_permute(g, q, 0, 2, 1, 3);
                auto * kp = ggml_cast(g, ggml_permute(g, kk, 0, 2, 1, 3), GGML_TYPE_F16);
                auto * vp = ggml_cast(g, ggml_permute(g, view(2), 0, 2, 1, 3), GGML_TYPE_F16);
                attended = ggml_flash_attn_ext(g, qp, kp, vp, nullptr, scale, 0, 0);
                ggml_prec_set_acc(attended, GGML_PREC_F32);
                attended = ggml_reshape_2d(g, attended, c.hidden, n);
            } else {
                auto * qp = ggml_permute(g, q, 0, 2, 1, 3);
                auto * kp = ggml_permute(g, kk, 0, 2, 1, 3);
                auto * scores = ggml_mul_mat(g, kp, qp);
                ggml_prec_set_acc(scores, GGML_PREC_F32);
                scores = ggml_soft_max_ext(g, scores, nullptr, scale, 0);
                auto * vt = ggml_cont(g, ggml_permute(g, view(2), 1, 2, 0, 3));
                auto * out = ggml_mul_mat(g, vt, scores);
                attended = ggml_cont_2d(g, ggml_permute(g, out, 0, 2, 1, 3), c.hidden, n);
            }
            x = ggml_add(g, x, mm(p + "attn_out", attended));
            h = layer_norm(x, p + "ln2");
            x = ggml_add(g, x, mm(p + "ffn_down", gelu_tanh(mm(p + "ffn_up", h))));
        }
        x = layer_norm(x, "v.post_ln");
        // Merger: each 2x2 block of consecutive patches becomes one token
        // (exact GELU, nn.GELU in the reference).
        x = ggml_reshape_2d(g, x, 4 * int64_t(c.hidden), n / 4);
        auto * out = mm("mm.2", ggml_gelu_erf(g, mm("mm.0", x)));
        ggml_set_output(out);
        ggml_build_forward_expand(graph, out);
        bool supported = true;
        std::string missing;
        for (int i = 0; i < ggml_graph_n_nodes(graph) && supported; ++i) {
            auto * node = ggml_graph_node(graph, i);
            if (!ggml_backend_supports_op(backend_.get(), node)) { supported = false; missing = ggml_op_desc(node); }
        }
        if (!supported) {
            if (flash_ && missing.find("FLASH_ATTN") != std::string::npos) { flash_ = false; continue; }
            throw std::runtime_error(std::string(ggml_backend_name(backend_.get())) + " does not support " + missing + " in the vision tower; no CPU fallback");
        }
        if (!flash_ && int64_t(n) * n * c.heads > (int64_t(1) << 31))
            throw std::runtime_error("Image too large for the vision tower without flash attention; lower --image-max-tokens");
        if (!allocator_) allocator_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_.get())));
        if (!allocator_ || !ggml_gallocr_alloc_graph(allocator_.get(), graph))
            throw std::runtime_error(std::string("Cannot allocate the vision graph on ") + ggml_backend_name(backend_.get()) +
                                     " (" + std::to_string(n) + " patches); lower --image-max-tokens");
        ggml_backend_tensor_set(patches, in.patches.data(), 0, in.patches.size() * sizeof(float));
        ggml_backend_tensor_set(positions, in.positions.data(), 0, in.positions.size() * sizeof(float));
        ggml_backend_tensor_set(rope, in.rope.data(), 0, in.rope.size() * sizeof(int32_t));
        ProfileScope compute(&profiler, "graph", "compute", "vision", -1, -1, ggml_backend_name(backend_.get()));
        if (ggml_backend_graph_compute(backend_.get(), graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("Vision graph execution failed");
        compute.finish();
        ImageEmbedding result;
        result.grid_width = pw / 2; result.grid_height = ph / 2;
        result.pixel_width = in.width; result.pixel_height = in.height;
        result.values.resize(size_t(ggml_nelements(out)));
        ggml_backend_tensor_get(out, result.values.data(), 0, ggml_nbytes(out));
        for (float v : result.values) if (!std::isfinite(v)) throw std::runtime_error("Non-finite vision tower output");
        result.hash = image.hash();
        ++stats.images_encoded;
        stats.image_tokens += uint64_t(result.tokens());
        stats.vision_ms += std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        return result;
    }
    throw std::logic_error("Vision graph construction did not converge");
}

VisionInfo VisionEncoder::info() const {
    VisionInfo info;
    info.enabled = true;
    info.placement = cpu_ ? "cpu" : "gpu";
    info.device = ggml_backend_name(backend_.get());
    info.layers = config_.layers; info.hidden = config_.hidden; info.heads = config_.heads; info.ffn = config_.ffn;
    info.patch_size = config_.patch; info.merge = config_.merge; info.output = config_.output;
    info.min_tokens = config_.min_tokens; info.max_tokens = config_.max_tokens;
    info.parameters = parameters_;
    info.weight_bytes = ggml_backend_buffer_get_size(weights_.get());
    return info;
}

void VisionEncoder::placements(std::vector<TensorPlacement> & out) const {
    const std::string device = ggml_backend_name(backend_.get());
    const std::string memory = ggml_backend_buffer_name(weights_.get());
    for (const auto & name : order_) {
        TensorPlacement t;
        t.name = name; t.component = "vision"; t.role = "weight";
        if (name.starts_with("v.blk.")) t.group = "vision.blk." + name.substr(6, name.find('.', 6) - 6);
        else if (name.starts_with("mm.")) t.group = "vision.merger";
        else if (name.starts_with("v.post_ln")) t.group = "vision.merger";
        else t.group = "vision.embedding";
        if (name == "v.position_embd.weight") {
            // Interpolated on the host while preprocessing each image.
            t.type = "f32"; t.shape = {config_.hidden, int64_t(config_.position_side) * config_.position_side};
            t.bytes = position_table_.size() * sizeof(float); t.resident = true; t.memory = "CPU"; t.compute = "CPU";
        } else {
            const auto * tensor = tensors_.at(name);
            t.type = ggml_type_name(tensor->type);
            for (int i = 0; i < ggml_n_dims(tensor); ++i) t.shape.push_back(tensor->ne[i]);
            t.bytes = ggml_nbytes(tensor); t.resident = true; t.memory = memory; t.compute = device;
        }
        out.push_back(std::move(t));
    }
}

} // namespace tidal
