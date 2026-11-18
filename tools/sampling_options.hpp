#pragma once
#include "args.hpp"
#include "tidal/tidal.hpp"
#include <cmath>
#include <ostream>

inline std::vector<std::string> sampling_options(std::vector<std::string> names) {
    for (const char * name : {"--sampler", "--temp", "--top-k", "--top-p", "--min-p", "--repeat-penalty",
                              "--frequency-penalty", "--presence-penalty", "--penalty-last-n", "--seed", "--logits"}) names.emplace_back(name);
    return names;
}
inline tidal::SamplingConfig sampling_config(const Args & args) {
    tidal::SamplingConfig c;
    const auto backend = args.get("--sampler", "device");
    if (backend == "cpu") c.backend = tidal::SamplingBackend::CPU;
    else if (backend != "device") throw std::invalid_argument("--sampler must be device or cpu");
    auto real = [&](const char * name, float fallback) {
        if (!args.contains(name)) return fallback;
        auto s = args.get(name); float value = 0;
        auto [end, error] = std::from_chars(s.data(), s.data() + s.size(), value);
        if (error != std::errc{} || end != s.data() + s.size() || !std::isfinite(value))
            throw std::invalid_argument(std::string("Invalid number for ") + name);
        return value;
    };
    c.temperature = real("--temp", 0); c.top_k = args.integer("--top-k", 0);
    c.top_p = real("--top-p", 1); c.min_p = real("--min-p", 0);
    c.repeat_penalty = real("--repeat-penalty", 1);
    c.frequency_penalty = real("--frequency-penalty", 0); c.presence_penalty = real("--presence-penalty", 0);
    c.penalty_last_n = args.integer("--penalty-last-n", 64);
    const auto seed = args.get("--seed", "0");
    const auto [end, error] = std::from_chars(seed.data(), seed.data() + seed.size(), c.seed);
    if (error != std::errc{} || end != seed.data() + seed.size()) throw std::invalid_argument("Invalid unsigned 64-bit seed");
    c.validate(); return c;
}
inline bool capture_logits(const Args & args, bool fallback) {
    const int value = args.integer("--logits", fallback ? 1 : 0);
    if (value != 0 && value != 1) throw std::invalid_argument("--logits must be 0 or 1");
    return value != 0;
}
inline void sampling_help(std::ostream & out) {
    out << "  [--sampler device|cpu] [--temp 0] [--top-k 0] [--top-p 1] [--min-p 0]\n"
           "  [--repeat-penalty 1] [--frequency-penalty 0] [--presence-penalty 0]\n"
           "  [--penalty-last-n 64] [--seed 0] [--logits 0|1]\n"
           "Temperature 0 is greedy. Device uses the model backend. CPU is an explicit comparison path.\n";
}
inline void sampling_json(std::ostream & out, const tidal::SamplingConfig & c) {
    out << "{\"backend\":\"" << (c.backend == tidal::SamplingBackend::CPU ? "cpu" : "device")
        << "\",\"temperature\":" << c.temperature << ",\"top_k\":" << c.top_k
        << ",\"top_p\":" << c.top_p << ",\"min_p\":" << c.min_p << ",\"repeat_penalty\":" << c.repeat_penalty
        << ",\"frequency_penalty\":" << c.frequency_penalty << ",\"presence_penalty\":" << c.presence_penalty
        << ",\"penalty_last_n\":" << c.penalty_last_n << ",\"seed\":" << c.seed << '}';
}
inline void sampling_stats(std::ostream & out, const tidal::Statistics & s) {
    out << ",\"sampling_calls\":" << s.sampling_calls << ",\"sampling_ms\":" << s.sampling_ms
        << ",\"sampling_device_to_host_bytes\":" << s.sampling_device_to_host_bytes
        << ",\"sampling_host_to_device_bytes\":" << s.sampling_host_to_device_bytes;
}
