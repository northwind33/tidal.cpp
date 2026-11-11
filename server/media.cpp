#include "media.hpp"

#include <stdexcept>

namespace tidal::server {

std::string base64_decode(std::string_view text) {
    std::string out;
    out.reserve(text.size() / 4 * 3);
    uint32_t buffer = 0;
    int bits = 0;
    size_t padding = 0;
    for (const char c : text) {
        int value;
        if (c >= 'A' && c <= 'Z') value = c - 'A';
        else if (c >= 'a' && c <= 'z') value = c - 'a' + 26;
        else if (c >= '0' && c <= '9') value = c - '0' + 52;
        else if (c == '+' || c == '-') value = 62;
        else if (c == '/' || c == '_') value = 63;
        else if (c == '=') { ++padding; continue; }
        else if (c == ' ' || c == '\n' || c == '\r' || c == '\t') continue;
        else throw std::invalid_argument("Invalid base64 image data");
        if (padding) throw std::invalid_argument("Invalid base64 image data: data after padding");
        buffer = buffer << 6 | uint32_t(value);
        bits += 6;
        if (bits >= 8) { bits -= 8; out.push_back(char((buffer >> bits) & 0xff)); }
    }
    if (padding > 2 || bits >= 6) throw std::invalid_argument("Invalid base64 image data: truncated");
    return out;
}

std::shared_ptr<const Image> image_from_base64(std::string_view data) {
    if (data.empty()) throw std::invalid_argument("Empty image data");
    return std::make_shared<const Image>(decode_image(base64_decode(data)));
}

std::shared_ptr<const Image> image_from_url(std::string_view url) {
    if (url.substr(0, 5) != "data:") {
        if (url.substr(0, 7) == "http://" || url.substr(0, 8) == "https://")
            throw std::invalid_argument("Remote image URLs are not fetched by this server; send the image as a data URI "
                                        "(data:image/png;base64,...) or base64 data");
        throw std::invalid_argument("Unsupported image URL: expected a data URI (data:image/...;base64,...)");
    }
    const auto comma = url.find(',');
    if (comma == std::string_view::npos) throw std::invalid_argument("Malformed data URI");
    const auto header = url.substr(5, comma - 5);
    if (header.size() < 7 || header.substr(header.size() - 7) != ";base64")
        throw std::invalid_argument("Image data URIs must be base64-encoded");
    return image_from_base64(url.substr(comma + 1));
}

} // namespace tidal::server
