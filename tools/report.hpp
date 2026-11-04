#pragma once
#include "tidal/tidal.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <map>
#include <ostream>
#include <string_view>

namespace report {
inline void string(std::ostream & out, std::string_view s) {
    out << '"';
    const char * hex = "0123456789abcdef";
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') out << '\\' << char(c);
        else if (c < 32) out << "\\u00" << hex[c >> 4] << hex[c & 15];
        else out << char(c);
    }
    out << '"';
}
template<class T> void array(std::ostream & out, const std::vector<T> & values) {
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) { if (i) out << ','; out << values[i]; }
    out << ']';
}
inline double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) return 0;
    std::sort(values.begin(), values.end());
    const double index = fraction * (values.size() - 1);
    const auto lo = size_t(index), hi = std::min(lo + 1, values.size() - 1);
    return values[lo] + (values[hi] - values[lo]) * (index - lo);
}
inline void distribution(std::ostream & out, const std::vector<double> & values) {
    if (values.empty()) { out << "null"; return; }
    double sum = 0, variance = 0;
    for (double v : values) sum += v;
    const double mean = sum / values.size();
    for (double v : values) variance += (v - mean) * (v - mean);
    out << "{\"count\":" << values.size() << ",\"mean\":" << mean
        << ",\"median\":" << percentile(values, 0.5) << ",\"p95\":" << percentile(values, 0.95)
        << ",\"min\":" << *std::min_element(values.begin(), values.end())
        << ",\"max\":" << *std::max_element(values.begin(), values.end())
        << ",\"stddev\":" << std::sqrt(variance / values.size()) << '}';
}
inline void config(std::ostream & out, const tidal::Config & c) {
    out << '{';
#define FIELD(f) out << "\"" #f "\":" << c.f << ','
    FIELD(hidden); FIELD(ffn); FIELD(vocab); FIELD(layers); FIELD(context); FIELD(branches);
    FIELD(rank); FIELD(q_heads); FIELD(kv_heads); FIELD(head_dim); FIELD(rope_dim);
    FIELD(gdn_key_heads); FIELD(gdn_value_heads); FIELD(gdn_dim); FIELD(conv_kernel);
    FIELD(index_heads); FIELD(index_dim); FIELD(block_size); FIELD(top_blocks); FIELD(eps);
#undef FIELD
    out << "\"rope_base\":" << c.rope_base;
    if (c.experts) out << ",\"experts\":" << c.experts << ",\"experts_used\":" << c.experts_used
        << ",\"expert_ffn\":" << c.expert_ffn << ",\"shared_ffn\":" << c.shared_ffn
        << ",\"norm_topk\":" << (c.norm_topk ? "true" : "false");
    if (!c.ple.layers.empty()) {
        const auto & p = c.ple;
        out << ",\"ple\":{\"ngram_size\":" << p.ngram_size << ",\"heads_per_ngram\":" << p.heads_per_ngram
            << ",\"embedding_dim\":" << p.embedding_dim << ",\"conv_kernel\":" << p.conv_kernel
            << ",\"eos_token_id\":" << p.eos_token_id << ",\"layers\":[";
        for (size_t i = 0; i < p.layers.size(); ++i) {
            if (i) out << ',';
            const auto & l = p.layers[i];
            out << "{\"layer\":" << l.layer << ",\"vocab\":" << l.vocab << ",\"multipliers\":";
            array(out, l.multipliers); out << ",\"offsets\":"; array(out, l.offsets);
            out << ",\"vocab_sizes\":"; array(out, l.vocab_sizes); out << '}';
        }
        out << "]}";
    }
    out << '}';
}
inline void stats(std::ostream & out, const tidal::Statistics & s) {
    out << "{\"graph_nodes\":" << s.device_graph_nodes
        << ",\"sparse_attention_calls\":" << s.sparse_attention_calls
        << ",\"d2h_bytes\":" << s.device_to_host_bytes << ",\"h2d_bytes\":" << s.host_to_device_bytes
        << ",\"graph_ms\":" << s.graph_ms << ",\"cpu_attention_ms\":" << s.attention_ms
        << ",\"transfer_ms\":" << s.transfer_ms << '}';
}
inline void event_args(std::ostream & out, const tidal::ProfileEvent & e) {
    out << "{\"phase\":"; string(out, e.phase);
    out << ",\"backend\":"; string(out, e.backend);
    out << ",\"position\":" << e.position << ",\"layer\":" << e.layer << ",\"head\":" << e.head
        << ",\"node\":" << e.node << ",\"bytes\":" << e.bytes << ",\"op\":"; string(out, e.op);
    out << ",\"dtype\":"; string(out, e.dtype);
    out << ",\"shape\":"; array(out, e.shape);
    out << ",\"input_types\":[";
    for (size_t i = 0; i < e.input_types.size(); ++i) { if (i) out << ','; string(out, e.input_types[i]); }
    out << "],\"input_shapes\":[";
    for (size_t i = 0; i < e.input_shapes.size(); ++i) { if (i) out << ','; array(out, e.input_shapes[i]); }
    out << "]}";
}
inline void events(std::ostream & out, const std::vector<tidal::ProfileEvent> & events, bool chrome) {
    out << '[';
    for (size_t i = 0; i < events.size(); ++i) {
        if (i) out << ',';
        const auto & e = events[i];
        out << "{\"name\":"; string(out, e.name);
        out << ",\"cat\":"; string(out, e.category);
        if (chrome) out << ",\"ph\":\"X\",\"pid\":1,\"tid\":" << (e.head < 0 ? 0 : e.head + 1);
        out << ",\"ts\":" << e.start_us << ",\"dur\":" << e.wall_us << ",\"args\":";
        event_args(out, e); out << '}';
    }
    out << ']';
}
struct Aggregate {
    tidal::ProfileEvent example;
    std::vector<double> durations;
    uint64_t bytes = 0;
    double total = 0;
};
inline std::vector<Aggregate> aggregate(const std::vector<tidal::ProfileEvent> & events) {
    std::map<std::string, Aggregate> groups;
    for (const auto & e : events) {
        // Position and shape may vary as the context grows; retain raw events
        // for context-dependent analysis. These are inclusive, overlapping totals.
        const auto key = e.category + "/" + e.phase + "/" + std::to_string(e.layer) + "/" + e.name + "/" + e.op;
        auto & a = groups[key];
        if (a.durations.empty()) a.example = e;
        a.durations.push_back(e.wall_us); a.total += e.wall_us; a.bytes += e.bytes;
    }
    std::vector<Aggregate> result;
    for (auto & [key, value] : groups) result.push_back(std::move(value));
    std::sort(result.begin(), result.end(), [](const auto & a, const auto & b) { return a.total > b.total; });
    return result;
}
inline void aggregates(std::ostream & out, const std::vector<Aggregate> & groups) {
    out << '[';
    for (size_t i = 0; i < groups.size(); ++i) {
        if (i) out << ',';
        const auto & a = groups[i];
        out << "{\"category\":"; string(out, a.example.category);
        out << ",\"name\":"; string(out, a.example.name);
        out << ",\"phase\":"; string(out, a.example.phase);
        out << ",\"layer\":" << a.example.layer << ",\"op\":"; string(out, a.example.op);
        out << ",\"total_us\":" << a.total << ",\"bytes\":" << a.bytes << ",\"wall_us\":";
        distribution(out, a.durations); out << '}';
    }
    out << ']';
}
} // namespace report
