#include "internal.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>

namespace tidal {
bool storage_type(ggml_type type) {
    if (type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16) return true;
    if (type < 0 || type >= GGML_TYPE_COUNT) return false;
    const auto * traits = ggml_get_type_traits(type);
    // Excludes removed types, integer tensors and Q8_1/Q8_K dot-product work
    // formats, which have no GGUF weight encoder/decoder in the pinned ggml.
    return traits->is_quantized && traits->to_float && traits->blck_size > 0;
}
std::vector<ggml_type> storage_types() {
    std::vector<ggml_type> result;
    for (int i = 0; i < GGML_TYPE_COUNT; ++i)
        if (storage_type(ggml_type(i))) result.push_back(ggml_type(i));
    return result;
}
ggml_type parse_storage_type(const std::string & name) {
    auto lower = [](std::string value) {
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return char(std::tolower(c)); });
        return value;
    };
    for (auto type : storage_types()) if (lower(ggml_type_name(type)) == lower(name)) return type;
    throw std::invalid_argument("Unsupported GGUF weight format: " + name + "; use --list-types 1");
}
void dequantize(ggml_type type, const void * data, float * values, int64_t count) {
    if (!storage_type(type) || count % ggml_blck_size(type)) throw std::runtime_error("Invalid dequantization request");
    if (type == GGML_TYPE_F32) std::memcpy(values, data, size_t(count) * sizeof(float));
    else ggml_get_type_traits(type)->to_float(data, values, count);
}
} // namespace tidal
