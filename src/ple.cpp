#include "internal.hpp"

#include <algorithm>
#include <limits>
#include <set>

namespace tidal {
void validate_ple(const Config & c) {
    const auto & p = c.ple;
    if (p.layers.empty()) return;
    if (p.ngram_size < 2 || p.ngram_size > 16 || p.heads_per_ngram < 1 || p.heads_per_ngram > 64 ||
        p.heads() > 64 || p.embedding_dim < 1 || p.embedding_dim > 4096 ||
        p.embedding_dim % p.heads() || p.embedding_dim % 32 ||
        p.conv_kernel < 1 || p.conv_kernel > 16 || p.eos_token_id < 0 || p.eos_token_id >= c.vocab)
        throw std::invalid_argument("Invalid PLE dimensions or EOS token");
    int previous = -1;
    for (const auto & l : p.layers) {
        if (l.layer <= previous || l.layer >= c.layers || (l.layer + 1) % 4 == 0)
            throw std::invalid_argument("PLE layers must be sorted, unique, zero-based GDN layers");
        previous = l.layer;
        if (l.vocab <= 0 || l.vocab > 1048576 ||
            int64_t(l.vocab) * (p.embedding_dim / p.heads()) > 64 * 1024 * 1024 ||
            l.multipliers.size() != size_t(p.ngram_size) ||
            l.offsets.size() != size_t(p.heads()) || l.vocab_sizes.size() != size_t(p.heads()))
            throw std::invalid_argument("Invalid PLE table or hash array length");
        // Reference hashes multiply signed int64 tokens without overflowing.
        // This bound preserves that behavior, including for imported constants.
        for (uint64_t m : l.multipliers)
            if (!m || m > uint64_t(INT64_MAX) / uint64_t(std::max(c.vocab - 1, 1)))
                throw std::invalid_argument("PLE multiplier exceeds signed int64 token product");
        uint64_t end = 0;
        for (int h = 0; h < p.heads(); ++h) {
            if (l.offsets[h] != end || !l.vocab_sizes[h] || l.vocab_sizes[h] > uint64_t(l.vocab) - end)
                throw std::invalid_argument("Invalid PLE head range");
            end += l.vocab_sizes[h];
        }
    }
}

void write_ple(gguf_context * f, const PLEConfig & p) {
    if (p.layers.empty()) return;
    std::vector<uint32_t> layers;
    for (const auto & l : p.layers) layers.push_back(uint32_t(l.layer));
    gguf_set_arr_data(f, "tidal.ple.layers", GGUF_TYPE_UINT32, layers.data(), layers.size());
    auto scalar = [&](const char * name, int value) {
        gguf_set_val_u32(f, (std::string("tidal.ple.") + name).c_str(), uint32_t(value));
    };
    scalar("ngram_size", p.ngram_size); scalar("heads_per_ngram", p.heads_per_ngram);
    scalar("embedding_dim", p.embedding_dim); scalar("conv_kernel", p.conv_kernel);
    scalar("eos_token_id", p.eos_token_id);
    for (const auto & l : p.layers) {
        const auto prefix = "tidal.ple." + std::to_string(l.layer) + ".";
        gguf_set_val_u32(f, (prefix + "vocab").c_str(), uint32_t(l.vocab));
        auto array = [&](const char * name, const std::vector<uint64_t> & values) {
            gguf_set_arr_data(f, (prefix + name).c_str(), GGUF_TYPE_UINT64, values.data(), values.size());
        };
        array("multipliers", l.multipliers); array("offsets", l.offsets); array("vocab_sizes", l.vocab_sizes);
    }
}

void read_ple(const gguf_context * f, Config & c) {
    auto & p = c.ple;
    std::set<std::string> consumed;
    auto key = [&](const std::string & name, gguf_type type) {
        auto id = gguf_find_key(f, name.c_str());
        if (id < 0 || gguf_get_kv_type(f, id) != type)
            throw std::runtime_error("Missing or wrongly typed PLE metadata: " + name);
        consumed.insert(name);
        return id;
    };
    auto scalar = [&](const std::string & name) {
        auto value = gguf_get_val_u32(f, key(name, GGUF_TYPE_UINT32));
        if (value > uint32_t(INT32_MAX)) throw std::runtime_error("PLE metadata overflow: " + name);
        return int(value);
    };
    auto array_id = [&](const std::string & name, gguf_type type, size_t max_count) {
        auto id = key(name, GGUF_TYPE_ARRAY);
        if (gguf_get_arr_type(f, id) != type || gguf_get_arr_n(f, id) > max_count)
            throw std::runtime_error("Invalid PLE metadata array: " + name);
        return id;
    };
    if (gguf_find_key(f, "tidal.ple.layers") >= 0) {
        const auto id = array_id("tidal.ple.layers", GGUF_TYPE_UINT32, 32);
        const auto count = gguf_get_arr_n(f, id);
        if (!count) throw std::runtime_error("Empty PLE layer list: omit the PLE metadata group instead");
        p.ngram_size = scalar("tidal.ple.ngram_size");
        p.heads_per_ngram = scalar("tidal.ple.heads_per_ngram");
        p.embedding_dim = scalar("tidal.ple.embedding_dim");
        p.conv_kernel = scalar("tidal.ple.conv_kernel");
        p.eos_token_id = scalar("tidal.ple.eos_token_id");
        auto * layers = static_cast<const uint32_t *>(gguf_get_arr_data(f, id));
        for (size_t i = 0; i < count; ++i) {
            if (layers[i] >= uint32_t(c.layers)) throw std::runtime_error("PLE layer out of range");
            PLELayer l;
            l.layer = int(layers[i]);
            const auto prefix = "tidal.ple." + std::to_string(l.layer) + ".";
            l.vocab = scalar(prefix + "vocab");
            auto array = [&](const char * name) {
                auto a = array_id(prefix + name, GGUF_TYPE_UINT64, 64);
                auto * values = static_cast<const uint64_t *>(gguf_get_arr_data(f, a));
                std::vector<uint64_t> result;
                for (size_t j = 0; j < gguf_get_arr_n(f, a); ++j) result.push_back(values[j]);
                return result;
            };
            l.multipliers = array("multipliers"); l.offsets = array("offsets"); l.vocab_sizes = array("vocab_sizes");
            p.layers.push_back(std::move(l));
        }
    }
    for (int64_t i = 0; i < gguf_get_n_kv(f); ++i) {
        std::string name = gguf_get_key(f, i);
        if (name.starts_with("tidal.ple.") && !consumed.contains(name))
            throw std::runtime_error("Unexpected or incomplete PLE metadata: " + name);
    }
}

std::vector<int32_t> ple_rows(const PLEConfig & p, const PLELayer & l,
                              int32_t token, const std::vector<int32_t> & previous) {
    std::vector<int32_t> result;
    result.reserve(p.heads());
    uint64_t mixed = uint64_t(token) * l.multipliers[0];
    bool cut = false;
    for (int shift = 1; shift < p.ngram_size; ++shift) {
        int32_t t = size_t(shift) <= previous.size() ? previous[previous.size() - shift] : p.eos_token_id;
        cut = cut || t == p.eos_token_id;
        if (cut) t = p.eos_token_id;
        mixed ^= uint64_t(t) * l.multipliers[shift];
        for (int g = 0; g < p.heads_per_ngram; ++g) {
            const int h = (shift - 1) * p.heads_per_ngram + g;
            result.push_back(int32_t(mixed % l.vocab_sizes[h] + l.offsets[h]));
        }
    }
    return result;
}
} // namespace tidal
