#pragma once
#include "args.hpp"
#include "tidal/tidal.hpp"
#include <ostream>

inline std::vector<std::string> prefill_options(std::vector<std::string> names) {
    for (const char * name : {"--prefill-precision", "--prefill-tile", "--prefill-memory-mib", "--prefill-reserve-mib"}) names.emplace_back(name);
    return names;
}
inline void configure_prefill(const Args & args, tidal::Options & options) {
    const auto precision = args.get("--prefill-precision", "fp16");
    if (precision == "fp16") options.prefill_precision = tidal::PrefillPrecision::FP16;
    else if (precision == "bf16") options.prefill_precision = tidal::PrefillPrecision::BF16;
    else throw std::invalid_argument("--prefill-precision must be fp16 or bf16");
    options.prefill_attention_tile = args.integer("--prefill-tile", 128);
    const int limit = args.integer("--prefill-memory-mib", 0), reserve = args.integer("--prefill-reserve-mib", 64);
    if (options.prefill_attention_tile < 1 || options.prefill_attention_tile > 4096 || limit < 0 || reserve < 0)
        throw std::invalid_argument("Invalid prefill tile or memory budget");
    options.prefill_memory_limit = uint64_t(limit)*1024*1024;
    options.prefill_memory_reserve = uint64_t(reserve)*1024*1024;
}
inline void prefill_help(std::ostream & out) {
    out << "  [--prefill-precision fp16|bf16] [--prefill-tile 128]\n"
           "  [--prefill-memory-mib 0] [--prefill-reserve-mib 64]\n"
           "CUDA prefill processes all prompt tokens in each layer before advancing.\n"
           "The tile limits queries within a layer. Memory 0 uses live free VRAM.\n";
}
inline void prefill_json(std::ostream & out, const tidal::Options & o) {
    out << "{\"execution\":\"" << (o.backend == "cuda" ? "layer_major" : "sequential_reference")
        << "\",\"attention_backend\":\"" << (o.backend == "cuda" ? "CUDA" : "CPU")
        << "\",\"precision\":\"" << (o.prefill_precision == tidal::PrefillPrecision::FP16 ? "fp16" : "bf16")
        << "\",\"attention_tile\":" << o.prefill_attention_tile
        << ",\"memory_limit_bytes\":" << o.prefill_memory_limit
        << ",\"memory_reserve_bytes\":" << o.prefill_memory_reserve << '}';
}
inline void prefill_stats(std::ostream & out, const tidal::Statistics & s) {
    out << ",\"prefill_tokens\":" << s.prefill_tokens << ",\"prefill_layers\":" << s.prefill_layers
        << ",\"prefill_gpu_attention_calls\":" << s.prefill_gpu_attention_calls
        << ",\"prefill_kv_to_host_bytes\":" << s.prefill_kv_to_host_bytes
        << ",\"prefill_weight_evictions\":" << s.prefill_weight_evictions
        << ",\"prefill_weight_reloads\":" << s.prefill_weight_reloads
        << ",\"prefill_weight_prefetches\":" << s.prefill_weight_prefetches
        << ",\"prefill_weight_reload_bytes\":" << s.prefill_weight_reload_bytes
        << ",\"prefill_peak_workspace_bytes\":" << s.prefill_peak_workspace_bytes;
}
