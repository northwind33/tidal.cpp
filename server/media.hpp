#pragma once
// Image inputs of API requests: data URIs and base64 payloads decoded into
// tidal::Image (PNG, JPEG, BMP, PPM).
#include "tidal/tidal.hpp"

#include <memory>
#include <string>
#include <string_view>

namespace tidal::server {

// The text the Qwen3.5/Qwen3.8 chat templates render for an image content part;
// its <|image_pad|> expands to the image's placeholder tokens after tokenizing.
inline constexpr const char * image_marker = "<|vision_start|><|image_pad|><|vision_end|>";
// llama.cpp's media marker in /completion prompts with multimodal_data.
inline constexpr const char * media_marker = "<__media__>";

// Strict base64 (standard or URL alphabet, optional padding, whitespace ignored).
std::string base64_decode(std::string_view text);
// "data:image/...;base64,...". Throws std::invalid_argument for remote URLs:
// the server does not fetch them.
std::shared_ptr<const Image> image_from_url(std::string_view url);
std::shared_ptr<const Image> image_from_base64(std::string_view data);

} // namespace tidal::server
