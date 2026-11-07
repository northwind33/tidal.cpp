#pragma once
#include "args.hpp"
#include "tidal/tidal.hpp"
#include <ostream>

inline std::vector<std::string> prefill_options(std::vector<std::string> names) {
    for (const char * name : {"--prefill-precision", "--prefill-chunk", "--prefill-tile", "--prefill-memory-mib", "--prefill-reserve-mib"}) names.emplace_back(name);
    return names;
}
inline void configure_prefill(const Args & args, tidal::Options & options) {
    const auto precision = args.get("--prefill-precision", "fp16");
    if (precision == "fp16") options.prefill_precision = tidal::PrefillPrecision::FP16;
    else if (precision == "bf16") options.prefill_precision = tidal::PrefillPrecision::BF16;
    else throw std::invalid_argument("--prefill-precision must be fp16 or bf16");
    auto size = [&](const char * flag) {
        const auto value = args.get(flag, "auto");
        if (value == "auto") return 0;
        const int n = number(value);
        if (n < 1) throw std::invalid_argument(std::string(flag)+" must be auto or a positive integer");
        return n;
    };
    options.prefill_chunk_tokens = size("--prefill-chunk");
    options.prefill_attention_tile = size("--prefill-tile");
    const int limit = args.integer("--prefill-memory-mib", 0);
    const auto reserve_text = args.get("--prefill-reserve-mib", "auto");
    const int reserve = reserve_text == "auto" ? 0 : number(reserve_text);
    if (options.prefill_attention_tile > 4096 || limit < 0 || reserve < 0)
        throw std::invalid_argument("Invalid prefill tile or memory budget");
    options.prefill_memory_limit = uint64_t(limit)*1024*1024;
    options.prefill_memory_reserve = reserve_text == "auto" ? UINT64_MAX : uint64_t(reserve)*1024*1024;
}
inline void prefill_help(std::ostream & out) {
    out << "  [--prefill-precision fp16|bf16] [--prefill-chunk auto|N] [--prefill-tile auto|N]\n"
           "  [--prefill-memory-mib 0] [--prefill-reserve-mib auto|MiB]\n"
           "CUDA prefill processes all prompt tokens in each layer before advancing.\n"
           "Chunks bound intermediate tensors; tiles bound attention queries (1-4096).\n"
           "Auto sizes shrink on pre-compute allocation failure; explicit sizes fail.\n"
           "Memory 0 uses live free VRAM; automatic reserve leaves kernel headroom.\n";
}
inline void prefill_json(std::ostream & out, const tidal::Options & o) {
    out << "{\"execution\":\"" << (o.backend == "cuda" ? "layer_major" : "sequential_reference")
        << "\",\"attention_backend\":\"" << (o.backend == "cuda" ? "CUDA" : "CPU")
        << "\",\"precision\":\"" << (o.prefill_precision == tidal::PrefillPrecision::FP16 ? "fp16" : "bf16")
        << "\",\"chunk_tokens\":" << o.prefill_chunk_tokens
        << ",\"chunk_mode\":\"" << (o.prefill_chunk_tokens ? "explicit" : "auto") << '"'
        << ",\"attention_tile\":" << o.prefill_attention_tile
        << ",\"attention_tile_mode\":\"" << (o.prefill_attention_tile ? "explicit" : "auto") << '"'
        << ",\"memory_limit_bytes\":" << o.prefill_memory_limit
        << ",\"memory_reserve_bytes\":";
    if (o.prefill_memory_reserve == UINT64_MAX) out << "null";
    else out << o.prefill_memory_reserve;
    out << ",\"memory_reserve_mode\":\"" << (o.prefill_memory_reserve == UINT64_MAX ? "auto" : "explicit") << "\"}";
}
inline void prefill_stats(std::ostream & out, const tidal::Statistics & s) {
    out << ",\"prefill_tokens\":" << s.prefill_tokens << ",\"prefill_layers\":" << s.prefill_layers
        << ",\"prefill_gpu_attention_calls\":" << s.prefill_gpu_attention_calls
        << ",\"prefill_kv_to_host_bytes\":" << s.prefill_kv_to_host_bytes
        << ",\"prefill_weight_evictions\":" << s.prefill_weight_evictions
        << ",\"prefill_weight_reloads\":" << s.prefill_weight_reloads
        << ",\"prefill_weight_prefetches\":" << s.prefill_weight_prefetches
        << ",\"prefill_weight_reload_bytes\":" << s.prefill_weight_reload_bytes
        << ",\"prefill_peak_workspace_bytes\":" << s.prefill_peak_workspace_bytes
        << ",\"prefill_chunks\":" << s.prefill_chunks
        << ",\"prefill_chunk_retries\":" << s.prefill_chunk_retries
        << ",\"prefill_chunk_min_tokens\":" << s.prefill_chunk_min_tokens
        << ",\"prefill_chunk_max_tokens\":" << s.prefill_chunk_max_tokens
        << ",\"prefill_attention_min_tile\":" << s.prefill_attention_min_tile
        << ",\"prefill_attention_max_tile\":" << s.prefill_attention_max_tile
        << ",\"prefill_reserve_bytes\":" << s.prefill_reserve_bytes;
}
