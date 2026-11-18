#include "json.hpp"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace tidal::server {

Json::Json(float value) {
    char buffer[40];
    const auto end = std::to_chars(buffer, buffer + sizeof buffer, value).ptr;
    double exact = value;
    std::from_chars(buffer, end, exact);
    value_ = exact;
}

const Json * Json::find(std::string_view key) const {
    if (!is_object()) return nullptr;
    for (const auto & member : members()) if (member.first == key) return &member.second;
    return nullptr;
}
const Json & Json::at(std::string_view key) const {
    static const Json null;
    const auto * value = find(key);
    return value ? *value : null;
}
Json & Json::operator[](std::string_view key) {
    if (is_null()) value_ = Object{};
    auto & object = members();
    for (auto & member : object) if (member.first == key) return member.second;
    object.emplace_back(std::string(key), Json());
    return object.back().second;
}
void Json::push(Json value) {
    if (is_null()) value_ = Array{};
    items().push_back(std::move(value));
}

void append_utf8(std::string & out, uint32_t c) {
    if (c < 0x80) out += char(c);
    else if (c < 0x800) { out += char(0xC0 | (c >> 6)); out += char(0x80 | (c & 0x3F)); }
    else if (c < 0x10000) { out += char(0xE0 | (c >> 12)); out += char(0x80 | ((c >> 6) & 0x3F)); out += char(0x80 | (c & 0x3F)); }
    else { out += char(0xF0 | (c >> 18)); out += char(0x80 | ((c >> 12) & 0x3F)); out += char(0x80 | ((c >> 6) & 0x3F)); out += char(0x80 | (c & 0x3F)); }
}

// Length of the well-formed sequence at `text[i]`, or 0 if there is none. Then
// `bad` is the length of the ill-formed part to replace as a whole (the longest
// prefix of a well-formed sequence, at least one byte) and `partial` tells
// whether the text ended before the sequence could be judged.
static size_t utf8_sequence(std::string_view text, size_t i, size_t * bad = nullptr, bool * partial = nullptr) {
    const uint8_t lead = uint8_t(text[i]);
    if (bad) *bad = 1;
    if (partial) *partial = false;
    if (lead < 0x80) return 1;
    size_t length; uint8_t low = 0x80, high = 0xBF;
    if (lead >= 0xC2 && lead <= 0xDF) length = 2;
    else if (lead >= 0xE0 && lead <= 0xEF) { length = 3; if (lead == 0xE0) low = 0xA0; if (lead == 0xED) high = 0x9F; }
    else if (lead >= 0xF0 && lead <= 0xF4) { length = 4; if (lead == 0xF0) low = 0x90; if (lead == 0xF4) high = 0x8F; }
    else return 0;
    for (size_t k = 1; k < length; ++k) {
        if (bad) *bad = k;
        if (i + k >= text.size()) { if (partial) *partial = true; return 0; }
        const uint8_t c = uint8_t(text[i + k]);
        if (c < (k == 1 ? low : 0x80) || c > (k == 1 ? high : 0xBF)) return 0;
    }
    return length;
}

std::string sanitize_utf8(std::string_view text) {
    std::string out; out.reserve(text.size());
    for (size_t i = 0; i < text.size();) {
        size_t bad = 1;
        const size_t n = utf8_sequence(text, i, &bad);
        if (n) { out.append(text, i, n); i += n; }
        else { out += "\xEF\xBF\xBD"; i += bad; }
    }
    return out;
}

size_t utf8_complete_prefix(std::string_view text) {
    // Only the last three bytes can start a sequence that the end cuts short.
    for (size_t back = 1; back <= 3 && back <= text.size(); ++back) {
        const size_t i = text.size() - back;
        if ((uint8_t(text[i]) & 0xC0) == 0x80) continue;
        bool partial = false;
        utf8_sequence(text, i, nullptr, &partial);
        return partial ? i : text.size();
    }
    return text.size();
}

static void write_string(std::string & out, std::string_view text) {
    out += '"';
    for (size_t i = 0; i < text.size();) {
        const uint8_t c = uint8_t(text[i]);
        if (c == '"') { out += "\\\""; ++i; }
        else if (c == '\\') { out += "\\\\"; ++i; }
        else if (c == '\n') { out += "\\n"; ++i; }
        else if (c == '\r') { out += "\\r"; ++i; }
        else if (c == '\t') { out += "\\t"; ++i; }
        else if (c == '\b') { out += "\\b"; ++i; }
        else if (c == '\f') { out += "\\f"; ++i; }
        else if (c < 0x20) {
            char buffer[8]; std::snprintf(buffer, sizeof buffer, "\\u%04x", c); out += buffer; ++i;
        } else {
            size_t bad = 1;
            const size_t n = utf8_sequence(text, i, &bad);
            if (n) { out.append(text, i, n); i += n; }
            else { out += "\xEF\xBF\xBD"; i += bad; }
        }
    }
    out += '"';
}

void Json::write(std::string & out, bool python) const {
    if (is_null()) out += "null";
    else if (is_bool()) out += boolean() ? "true" : "false";
    else if (is_int()) out += std::to_string(integer());
    else if (is_number()) {
        const double value = std::get<double>(value_);
        if (!std::isfinite(value)) { out += "null"; return; }
        char buffer[40];
        const auto end = std::to_chars(buffer, buffer + sizeof buffer, value).ptr;
        std::string_view text(buffer, size_t(end - buffer));
        out += text;
        if (text.find_first_of(".e") == std::string_view::npos) out += ".0";
    } else if (is_string()) write_string(out, string());
    else if (is_array()) {
        out += '[';
        bool first = true;
        for (const auto & item : items()) {
            if (!first) out += python ? ", " : ",";
            first = false; item.write(out, python);
        }
        out += ']';
    } else {
        out += '{';
        bool first = true;
        for (const auto & member : members()) {
            if (!first) out += python ? ", " : ",";
            first = false; write_string(out, member.first);
            out += python ? ": " : ":"; member.second.write(out, python);
        }
        out += '}';
    }
}
std::string Json::dump(bool python) const { std::string out; write(out, python); return out; }

namespace {
struct Parser {
    std::string_view text;
    size_t at = 0;
    int depth = 0;
    [[noreturn]] void fail(const char * message) const {
        throw JsonError(std::string("Invalid JSON at byte ") + std::to_string(at) + ": " + message);
    }
    void space() { while (at < text.size() && (text[at] == ' ' || text[at] == '\n' || text[at] == '\r' || text[at] == '\t')) ++at; }
    bool literal(std::string_view word) {
        if (text.substr(at, word.size()) != word) return false;
        at += word.size(); return true;
    }
    uint32_t hex4() {
        if (at + 4 > text.size()) fail("truncated \\u escape");
        uint32_t value = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = text[at++];
            value <<= 4;
            if (c >= '0' && c <= '9') value |= uint32_t(c - '0');
            else if (c >= 'a' && c <= 'f') value |= uint32_t(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') value |= uint32_t(c - 'A' + 10);
            else fail("invalid \\u escape");
        }
        return value;
    }
    std::string string() {
        ++at;
        std::string out;
        for (;;) {
            if (at >= text.size()) fail("unterminated string");
            const uint8_t c = uint8_t(text[at]);
            if (c == '"') { ++at; return out; }
            if (c < 0x20) fail("control character in string");
            if (c != '\\') {
                size_t bad = 1;
                const size_t n = utf8_sequence(text, at, &bad);
                if (n) { out.append(text, at, n); at += n; }
                else { out += "\xEF\xBF\xBD"; at += bad; }
                continue;
            }
            if (++at >= text.size()) fail("truncated escape");
            switch (text[at++]) {
                case '"': out += '"'; break; case '\\': out += '\\'; break; case '/': out += '/'; break;
                case 'b': out += '\b'; break; case 'f': out += '\f'; break; case 'n': out += '\n'; break;
                case 'r': out += '\r'; break; case 't': out += '\t'; break;
                case 'u': {
                    uint32_t code = hex4();
                    if (code >= 0xD800 && code <= 0xDBFF && text.substr(at, 2) == "\\u") {
                        const size_t saved = at;
                        at += 2;
                        const uint32_t low = hex4();
                        if (low >= 0xDC00 && low <= 0xDFFF) code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
                        else at = saved;
                    }
                    // A lone surrogate has no UTF-8 encoding.
                    if (code >= 0xD800 && code <= 0xDFFF) code = 0xFFFD;
                    append_utf8(out, code); break;
                }
                default: fail("invalid escape");
            }
        }
    }
    Json number() {
        const size_t start = at;
        if (text[at] == '-') ++at;
        const auto digits = [&] {
            const size_t first = at;
            while (at < text.size() && text[at] >= '0' && text[at] <= '9') ++at;
            return at - first;
        };
        const size_t integer_start = at;
        if (!digits()) fail("invalid number");
        if (text[integer_start] == '0' && at - integer_start > 1) fail("leading zero");
        bool real = false;
        if (at < text.size() && text[at] == '.') { real = true; ++at; if (!digits()) fail("invalid fraction"); }
        if (at < text.size() && (text[at] == 'e' || text[at] == 'E')) {
            real = true; ++at;
            if (at < text.size() && (text[at] == '+' || text[at] == '-')) ++at;
            if (!digits()) fail("invalid exponent");
        }
        const char * first = text.data() + start, * last = text.data() + at;
        if (!real) {
            int64_t value = 0;
            const auto result = std::from_chars(first, last, value);
            if (result.ec == std::errc{} && result.ptr == last) return Json(value);
        }
        double value = 0;
        const auto result = std::from_chars(first, last, value);
        if (result.ec == std::errc::result_out_of_range) return Json(first[0] == '-' ? -HUGE_VAL : HUGE_VAL);
        if (result.ec != std::errc{} || result.ptr != last) fail("invalid number");
        return Json(value);
    }
    Json value() {
        space();
        if (at >= text.size()) fail("unexpected end");
        const char c = text[at];
        if (c == '"') return Json(string());
        if (c == '{' || c == '[') {
            if (++depth > 256) fail("nesting too deep");
            ++at;
            Json result = c == '{' ? Json::object() : Json::array();
            const char close = c == '{' ? '}' : ']';
            space();
            if (at < text.size() && text[at] == close) { ++at; --depth; return result; }
            for (;;) {
                if (c == '{') {
                    space();
                    if (at >= text.size() || text[at] != '"') fail("expected object key");
                    auto key = string();
                    space();
                    if (at >= text.size() || text[at] != ':') fail("expected ':'");
                    ++at;
                    // The last duplicate wins, as in most parsers.
                    result[key] = value();
                } else result.push(value());
                space();
                if (at >= text.size()) fail("unexpected end");
                if (text[at] == ',') { ++at; continue; }
                if (text[at] == close) { ++at; --depth; return result; }
                fail("expected ',' or closing bracket");
            }
        }
        if (literal("true")) return Json(true);
        if (literal("false")) return Json(false);
        if (literal("null")) return Json();
        if (c == '-' || (c >= '0' && c <= '9')) return number();
        fail("unexpected character");
    }
};
} // namespace

Json Json::parse(std::string_view text) {
    Parser parser{text};
    Json result = parser.value();
    parser.space();
    if (parser.at != text.size()) parser.fail("trailing characters");
    return result;
}

} // namespace tidal::server
