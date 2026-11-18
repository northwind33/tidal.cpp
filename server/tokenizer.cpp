#include "tokenizer.hpp"
#include "json.hpp"
#include "unicode_tables.hpp"

#include "gguf.h"

#include <algorithm>
#include <array>
#include <memory>
#include <queue>
#include <stdexcept>

namespace tidal::server {
namespace {

enum : uint8_t { Letter = 1, Mark = 2, Number = 4, Space = 8, Newline = 16 };

template<size_t N> bool in_ranges(const unicode_tables::Range (&ranges)[N], uint32_t c) {
    const auto * it = std::upper_bound(std::begin(ranges), std::end(ranges), c,
        [](uint32_t value, const unicode_tables::Range & range) { return value < range.first; });
    return it != std::begin(ranges) && c <= (it - 1)->last;
}
// Unicode White_Space, the meaning of \s in the tokenizer's pattern.
bool is_space(uint32_t c) {
    return (c >= 0x09 && c <= 0x0D) || c == 0x20 || c == 0x85 || c == 0xA0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}
uint8_t classify(uint32_t c) {
    static const std::array<uint8_t, 128> ascii = [] {
        std::array<uint8_t, 128> table{};
        for (uint32_t i = 0; i < 128; ++i) {
            if ((i >= 'a' && i <= 'z') || (i >= 'A' && i <= 'Z')) table[i] = Letter;
            else if (i >= '0' && i <= '9') table[i] = Number;
            else if (is_space(i)) table[i] = Space | ((i == '\r' || i == '\n') ? Newline : 0);
        }
        return table;
    }();
    if (c < 128) return ascii[c];
    if (in_ranges(unicode_tables::letters, c)) return Letter;
    if (in_ranges(unicode_tables::marks, c)) return Mark;
    if (in_ranges(unicode_tables::numbers, c)) return Number;
    return is_space(c) ? Space : 0;
}

// GPT-2 byte-level alphabet: printable Latin-1 bytes map to themselves, the
// rest to U+0100 onwards, so that every byte is a visible character.
const std::array<uint32_t, 256> & byte_alphabet() {
    static const std::array<uint32_t, 256> table = [] {
        std::array<uint32_t, 256> map{};
        uint32_t next = 256;
        for (uint32_t b = 0; b < 256; ++b) {
            const bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174);
            map[b] = printable ? b : next++;
        }
        return map;
    }();
    return table;
}

struct Decoded { uint32_t code; uint32_t length; };
// `text` must be valid UTF-8.
Decoded decode_utf8(std::string_view text, size_t i) {
    const uint8_t lead = uint8_t(text[i]);
    if (lead < 0x80) return {lead, 1};
    const auto tail = [&](size_t k) { return uint32_t(uint8_t(text[i + k]) & 0x3F); };
    if (lead < 0xE0) return {(uint32_t(lead & 0x1F) << 6) | tail(1), 2};
    if (lead < 0xF0) return {(uint32_t(lead & 0x0F) << 12) | (tail(1) << 6) | tail(2), 3};
    return {(uint32_t(lead & 0x07) << 18) | (tail(1) << 12) | (tail(2) << 6) | tail(3), 4};
}

struct GGUF {
    gguf_context * f;
    explicit GGUF(const std::string & path) : f(gguf_init_from_file(path.c_str(), {true, nullptr})) {
        if (!f) throw std::runtime_error("Cannot read GGUF metadata: " + path);
    }
    ~GGUF() { gguf_free(f); }
    GGUF(const GGUF &) = delete;
    int64_t key(const char * name, gguf_type type) const {
        const auto id = gguf_find_key(f, name);
        return id >= 0 && gguf_get_kv_type(f, id) == type ? id : -1;
    }
    std::string text(const char * name) const {
        const auto id = key(name, GGUF_TYPE_STRING);
        return id < 0 ? std::string() : std::string(gguf_get_val_str(f, id));
    }
    std::optional<double> number(const char * name) const {
        const auto id = gguf_find_key(f, name);
        if (id < 0) return std::nullopt;
        switch (gguf_get_kv_type(f, id)) {
            case GGUF_TYPE_FLOAT32: return gguf_get_val_f32(f, id);
            case GGUF_TYPE_FLOAT64: return gguf_get_val_f64(f, id);
            case GGUF_TYPE_UINT32: return gguf_get_val_u32(f, id);
            case GGUF_TYPE_INT32: return gguf_get_val_i32(f, id);
            case GGUF_TYPE_UINT64: return double(gguf_get_val_u64(f, id));
            case GGUF_TYPE_INT64: return double(gguf_get_val_i64(f, id));
            default: return std::nullopt;
        }
    }
    int64_t strings(const char * name) const {
        const auto id = key(name, GGUF_TYPE_ARRAY);
        return id >= 0 && gguf_get_arr_type(f, id) == GGUF_TYPE_STRING ? id : -1;
    }
};

} // namespace

std::vector<size_t> Tokenizer::split(std::string_view text, bool marks) {
    // (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+ | \p{N} |
    //  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
    // matched as ordered alternatives. Without `marks` ("qwen2"), \p{M} is not
    // part of either class, so marks count as punctuation.
    std::vector<uint32_t> codes; std::vector<uint32_t> offsets; std::vector<uint8_t> kinds;
    codes.reserve(text.size()); offsets.reserve(text.size() + 1); kinds.reserve(text.size());
    for (size_t i = 0; i < text.size();) {
        const auto [code, length] = decode_utf8(text, i);
        codes.push_back(code); offsets.push_back(uint32_t(i)); kinds.push_back(classify(code));
        i += length;
    }
    offsets.push_back(uint32_t(text.size()));
    const size_t n = codes.size();
    const auto word = [&](size_t i) { return (kinds[i] & Letter) || (marks && (kinds[i] & Mark)); };
    const auto punctuation = [&](size_t i) {
        return !(kinds[i] & (Space | Letter | Number)) && !(marks && (kinds[i] & Mark));
    };
    const auto fold = [&](size_t i) -> uint32_t {
        const uint32_t c = codes[i];
        if (c >= 'A' && c <= 'Z') return c + 32;
        return c == 0x17F ? 's' : c; // LATIN SMALL LETTER LONG S folds to s.
    };
    std::vector<size_t> pieces;
    for (size_t i = 0; i < n;) {
        size_t end = i;
        if (codes[i] == '\'' && i + 1 < n) {
            const uint32_t a = fold(i + 1), b = i + 2 < n ? fold(i + 2) : 0;
            if (a == 's' || a == 't' || a == 'm' || a == 'd') end = i + 2;
            else if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) end = i + 3;
        }
        if (end == i) {
            size_t start = i;
            if (!(kinds[i] & (Newline | Letter | Number)) && i + 1 < n && word(i + 1)) start = i + 1;
            if (word(start)) { end = start + 1; while (end < n && word(end)) ++end; }
        }
        if (end == i && (kinds[i] & Number)) end = i + 1;
        if (end == i) {
            size_t start = i;
            if (codes[i] == ' ' && i + 1 < n && punctuation(i + 1)) start = i + 1;
            if (punctuation(start)) {
                end = start + 1;
                while (end < n && punctuation(end)) ++end;
                while (end < n && (kinds[end] & Newline)) ++end;
            }
        }
        if (end == i && (kinds[i] & Space)) {
            size_t run = i + 1, newline = (kinds[i] & Newline) ? i + 1 : 0;
            for (; run < n && (kinds[run] & Space); ++run) if (kinds[run] & Newline) newline = run + 1;
            if (newline) end = newline;               // \s*[\r\n]+
            else if (run == n || run - i == 1) end = run; // \s+(?!\S) at the end, or \s+
            else end = run - 1;                       // \s+(?!\S) leaves one for the next piece.
        }
        if (end == i) end = i + 1;
        pieces.push_back(offsets[end] - offsets[i]);
        i = end;
    }
    return pieces;
}

Tokenizer::Tokenizer(const std::string & gguf_path, int vocab) : vocab_(vocab) {
    if (vocab < 1) throw std::invalid_argument("Tokenizer vocabulary must be positive");
    GGUF g(gguf_path);
    metadata_.name = g.text("general.name");
    metadata_.architecture = g.text("general.architecture");
    metadata_.chat_template = g.text("tokenizer.chat_template");
    const auto real = [&](const char * name) -> std::optional<float> {
        const auto value = g.number(name);
        return value ? std::optional<float>(float(*value)) : std::nullopt;
    };
    metadata_.temperature = real("general.sampling.temp");
    metadata_.top_p = real("general.sampling.top_p");
    metadata_.min_p = real("general.sampling.min_p");
    metadata_.repeat_penalty = real("general.sampling.penalty_repeat");
    if (const auto top_k = g.number("general.sampling.top_k")) metadata_.top_k = int(*top_k);

    const auto tokens = g.strings("tokenizer.ggml.tokens");
    if (tokens < 0) {
        byte_fallback_ = true;
        return;
    }
    const auto model = g.text("tokenizer.ggml.model"), pre = g.text("tokenizer.ggml.pre");
    if (model != "gpt2") throw std::runtime_error("Unsupported tokenizer model '" + model + "': only byte-level BPE (gpt2) is implemented");
    if (pre == "qwen35") marks_ = true;
    else if (pre == "qwen2") marks_ = false;
    else throw std::runtime_error("Unsupported pre-tokenizer '" + pre + "': only qwen2 and qwen35 are implemented");
    const size_t count = gguf_get_arr_n(g.f, tokens);
    if (count != size_t(vocab)) throw std::runtime_error("Tokenizer vocabulary size differs from the model vocabulary");
    types_.assign(count, Normal);
    const auto types = g.key("tokenizer.ggml.token_type", GGUF_TYPE_ARRAY);
    if (types >= 0 && gguf_get_arr_type(g.f, types) == GGUF_TYPE_INT32 && gguf_get_arr_n(g.f, types) == count) {
        const auto * values = static_cast<const int32_t *>(gguf_get_arr_data(g.f, types));
        types_.assign(values, values + count);
    }
    // Inverse of the byte alphabet, to turn token text back into raw bytes.
    std::unordered_map<uint32_t, uint8_t> bytes;
    for (uint32_t b = 0; b < 256; ++b) bytes[byte_alphabet()[b]] = uint8_t(b);
    pieces_.resize(count); ids_.reserve(count * 2);
    for (size_t id = 0; id < count; ++id) {
        const std::string text = gguf_get_arr_str(g.f, tokens, id);
        ids_.emplace(text, int32_t(id));
        if (types_[id] != Normal) {
            pieces_[id] = text;
            if ((types_[id] == Control || types_[id] == UserDefined) && !text.empty())
                added_.push_back({text, int32_t(id), types_[id] == Control});
            continue;
        }
        const std::string valid = sanitize_utf8(text);
        for (size_t i = 0; i < valid.size();) {
            const auto [code, length] = decode_utf8(valid, i);
            const auto it = bytes.find(code);
            if (it != bytes.end()) pieces_[id] += char(it->second);
            else pieces_[id].append(valid, i, length);
            i += length;
        }
    }
    std::stable_sort(added_.begin(), added_.end(), [](const Added & a, const Added & b) { return a.text.size() > b.text.size(); });
    const auto merges = g.strings("tokenizer.ggml.merges");
    if (merges < 0) throw std::runtime_error("GGUF tokenizer has no BPE merges");
    const size_t merge_count = gguf_get_arr_n(g.f, merges);
    ranks_.reserve(merge_count * 2);
    for (size_t i = 0; i < merge_count; ++i) ranks_.emplace(gguf_get_arr_str(g.f, merges, i), int32_t(i));
    const auto token_id = [&](const char * name) {
        const auto value = g.number(name);
        return value && *value >= 0 && *value < double(count) ? int32_t(*value) : -1;
    };
    eos_ = token_id("tokenizer.ggml.eos_token_id");
    bos_ = token_id("tokenizer.ggml.bos_token_id");
    end_.assign(count, false);
    if (eos_ >= 0) end_[size_t(eos_)] = true;
    for (const char * name : {"<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end_of_text|>", "<|end|>", "<end_of_turn>"}) {
        const auto it = ids_.find(name);
        if (it != ids_.end() && types_[size_t(it->second)] == Control) end_[size_t(it->second)] = true;
    }
}

bool Tokenizer::is_control(int32_t token) const {
    return !byte_fallback_ && token >= 0 && token < vocab_ && (types_[size_t(token)] == Control || types_[size_t(token)] == Unused);
}
bool Tokenizer::is_end(int32_t token) const {
    return !byte_fallback_ && token >= 0 && token < vocab_ && end_[size_t(token)];
}
int32_t Tokenizer::added_token(std::string_view text) const {
    for (const auto & added : added_) if (added.text == text) return added.id;
    return -1;
}

std::string Tokenizer::piece(int32_t token, bool render_special) const {
    if (token < 0 || token >= vocab_) throw std::out_of_range("Token outside vocabulary");
    if (byte_fallback_) return token < 256 ? std::string(1, char(token)) : std::string("\xEF\xBF\xBD");
    if (!render_special && is_control(token)) return {};
    return pieces_[size_t(token)];
}
std::string Tokenizer::decode(const std::vector<int32_t> & tokens, bool render_special) const {
    std::string bytes;
    for (auto token : tokens) bytes += piece(token, render_special);
    return sanitize_utf8(bytes);
}

void Tokenizer::bpe(const std::string & mapped, std::vector<int32_t> & out) const {
    // Symbols are spans of `mapped` in a doubly linked list. Merges apply in
    // priority order, leftmost first within one priority.
    struct Symbol { uint32_t start, length; int32_t previous, next; };
    std::vector<Symbol> symbols;
    for (size_t i = 0; i < mapped.size();) {
        const uint32_t length = decode_utf8(mapped, i).length;
        symbols.push_back({uint32_t(i), length, int32_t(symbols.size()) - 1, int32_t(symbols.size()) + 1});
        i += length;
    }
    if (symbols.empty()) return;
    symbols.back().next = -1;
    struct Candidate {
        int32_t rank, left; uint32_t left_length, right_length;
        bool operator>(const Candidate & other) const { return rank != other.rank ? rank > other.rank : left > other.left; }
    };
    std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>> queue;
    std::string key;
    const auto consider = [&](int32_t left) {
        if (left < 0 || symbols[size_t(left)].next < 0) return;
        const auto & a = symbols[size_t(left)], & b = symbols[size_t(a.next)];
        key.assign(mapped, a.start, a.length); key += ' '; key.append(mapped, b.start, b.length);
        const auto it = ranks_.find(key);
        if (it != ranks_.end()) queue.push({it->second, left, a.length, b.length});
    };
    for (int32_t i = 0; i + 1 < int32_t(symbols.size()); ++i) consider(i);
    while (!queue.empty()) {
        const auto top = queue.top(); queue.pop();
        auto & a = symbols[size_t(top.left)];
        if (!a.length || a.length != top.left_length || a.next < 0) continue;
        auto & b = symbols[size_t(a.next)];
        if (b.length != top.right_length) continue;
        a.length += b.length; b.length = 0;
        a.next = b.next;
        if (a.next >= 0) symbols[size_t(a.next)].previous = top.left;
        consider(a.previous); consider(top.left);
    }
    for (int32_t i = 0; i >= 0; i = symbols[size_t(i)].next) {
        const auto & s = symbols[size_t(i)];
        const auto it = ids_.find(mapped.substr(s.start, s.length));
        if (it != ids_.end()) { out.push_back(it->second); continue; }
        // No token for this symbol: every byte-alphabet character is a token of
        // a well-formed vocabulary, so fall back to them one by one.
        for (size_t at = s.start; at < size_t(s.start) + s.length;) {
            const uint32_t length = decode_utf8(mapped, at).length;
            const auto single = ids_.find(mapped.substr(at, length));
            if (single != ids_.end()) out.push_back(single->second);
            at += length;
        }
    }
}

void Tokenizer::encode_text(std::string_view text, std::vector<int32_t> & out) const {
    const auto & alphabet = byte_alphabet();
    std::string mapped;
    size_t at = 0;
    for (const size_t length : split(text, marks_)) {
        const auto piece = text.substr(at, length);
        at += length;
        mapped.clear();
        for (const char c : piece) append_utf8(mapped, alphabet[uint8_t(c)]);
        const auto whole = ids_.find(mapped);
        if (whole != ids_.end() && types_[size_t(whole->second)] == Normal && piece.size() == 1) { out.push_back(whole->second); continue; }
        {
            std::lock_guard lock(cache_mutex_);
            const auto cached = cache_.find(mapped);
            if (cached != cache_.end()) { out.insert(out.end(), cached->second.begin(), cached->second.end()); continue; }
        }
        const size_t before = out.size();
        bpe(mapped, out);
        if (piece.size() <= 64) {
            std::lock_guard lock(cache_mutex_);
            if (cache_.size() < 200000) cache_.emplace(mapped, std::vector<int32_t>(out.begin() + long(before), out.end()));
        }
    }
}

std::vector<int32_t> Tokenizer::encode(std::string_view raw, bool parse_special) const {
    std::vector<int32_t> out;
    if (byte_fallback_) {
        out.reserve(raw.size());
        for (const char c : raw) out.push_back(int32_t(uint8_t(c)) % vocab_);
        return out;
    }
    const std::string text = sanitize_utf8(raw);
    const std::string_view view(text);
    size_t start = 0;
    for (size_t i = 0; i < view.size();) {
        const Added * match = nullptr;
        for (const auto & added : added_)
            if ((parse_special || !added.control) && view.compare(i, added.text.size(), added.text) == 0) { match = &added; break; }
        if (!match) { ++i; continue; }
        encode_text(view.substr(start, i - start), out);
        out.push_back(match->id);
        start = i += match->text.size();
    }
    encode_text(view.substr(start), out);
    return out;
}

} // namespace tidal::server
