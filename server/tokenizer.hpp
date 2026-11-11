#pragma once
// Text <-> token conversion from the tokenizer stored in a GGUF file.
//
// Supported: byte-level BPE ("gpt2" model) with the Qwen pre-tokenizers
// ("qwen2", "qwen35"). A GGUF without a tokenizer (the synthetic test models)
// gets a byte tokenizer: one token per UTF-8 byte.
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace tidal::server {

// GGUF metadata the server needs besides the vocabulary.
struct ModelMetadata {
    std::string name, architecture, chat_template;
    std::optional<float> temperature, top_p, min_p, repeat_penalty;
    std::optional<int> top_k;
};

class Tokenizer {
public:
    // `vocab` is the engine's vocabulary size; token ids must stay below it.
    Tokenizer(const std::string & gguf_path, int vocab);

    // `parse_special` also matches control tokens such as <|im_start|> in the
    // text; user-defined added tokens such as <think> are always matched.
    std::vector<int32_t> encode(std::string_view text, bool parse_special) const;
    // Raw bytes of one token; a multi-byte character can span several tokens.
    // Control tokens render as text only with `render_special`.
    std::string piece(int32_t token, bool render_special) const;
    // Concatenated pieces with invalid UTF-8 replaced by U+FFFD.
    std::string decode(const std::vector<int32_t> & tokens, bool render_special) const;
    // Id of an added (control or user-defined) token by its exact text, or -1.
    int32_t added_token(std::string_view text) const;
    bool is_control(int32_t token) const;
    // End-of-generation tokens: the declared EOS plus known end-of-turn markers.
    bool is_end(int32_t token) const;
    int32_t eos() const { return eos_; }
    int32_t bos() const { return bos_; }
    int size() const { return vocab_; }
    bool byte_fallback() const { return byte_fallback_; }
    const ModelMetadata & metadata() const { return metadata_; }

    // Pre-tokenization only: byte lengths of the consecutive pieces of `text`.
    // `marks` selects the "qwen35" pattern, which keeps combining marks in words.
    static std::vector<size_t> split(std::string_view text, bool marks);

private:
    enum TokenType : int32_t { Normal = 1, Unknown = 2, Control = 3, UserDefined = 4, Unused = 5, Byte = 6 };
    struct Added { std::string text; int32_t id; bool control; };

    int vocab_;
    bool byte_fallback_ = false, marks_ = true;
    int32_t eos_ = -1, bos_ = -1;
    ModelMetadata metadata_;
    std::vector<std::string> pieces_; // Decoded bytes per token.
    std::vector<int32_t> types_;
    std::vector<bool> end_;
    std::unordered_map<std::string, int32_t> ids_;   // Byte-level (mapped) token text.
    std::unordered_map<std::string, int32_t> ranks_; // "left right" merge -> priority.
    std::vector<Added> added_;                       // Longest first.
    mutable std::mutex cache_mutex_;
    mutable std::unordered_map<std::string, std::vector<int32_t>> cache_;

    void encode_text(std::string_view text, std::vector<int32_t> & out) const;
    void bpe(const std::string & mapped, std::vector<int32_t> & out) const;
};

} // namespace tidal::server
