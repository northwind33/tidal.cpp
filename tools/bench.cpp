#include "tidal/tidal.hpp"
#include "args.hpp"
#include "report.hpp"
#include "tidal-build-info.hpp"
#ifdef TIDAL_BENCH_CUDA
#include <cuda_runtime_api.h>
#endif
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>

namespace {
using Clock = std::chrono::steady_clock;
double ms(Clock::time_point start) { return std::chrono::duration<double, std::milli>(Clock::now() - start).count(); }
constexpr uint64_t hash_basis = 14695981039346656037ull;
void hash_bytes(uint64_t & hash, const void * data, size_t size) {
    const auto * p = static_cast<const unsigned char *>(data);
    for (size_t i = 0; i < size; ++i) { hash ^= p[i]; hash *= 1099511628211ull; }
}
std::string hex(uint64_t value) { std::ostringstream s; s << std::hex << std::setfill('0') << std::setw(16) << value; return s.str(); }
std::string file_hash(const std::string & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot read model: " + path);
    uint64_t hash = hash_basis;
    char buffer[65536];
    while (input.read(buffer, sizeof(buffer)) || input.gcount()) hash_bytes(hash, buffer, size_t(input.gcount()));
    if (!input.eof()) throw std::runtime_error("Failed reading model fingerprint");
    return hex(hash);
}
std::string cpu_name() {
    std::ifstream input("/proc/cpuinfo");
    for (std::string line; std::getline(input, line);)
        if (line.rfind("model name", 0) == 0 && line.find(':') != std::string::npos) return line.substr(line.find(':') + 2);
    return "unknown";
}
struct Run {
    double reset_ms = 0, prefill_ms = 0, decode_ms = 0;
    std::vector<double> prefill_latencies, decode_latencies;
    std::vector<int32_t> generated;
    std::vector<tidal::Step> trace;
    tidal::Statistics stats;
    std::string checksum;
    bool replay_greedy_matches = true;
};
Run execute(tidal::Engine & engine, const std::vector<int32_t> & prompt, int steps,
            const std::vector<int32_t> * replay = nullptr) {
    Run r;
    r.trace.reserve(prompt.size() + size_t(steps));
    r.generated.reserve(steps); r.prefill_latencies.reserve(prompt.size()); r.decode_latencies.reserve(steps);
    auto start = Clock::now(); engine.reset(); r.reset_ms = ms(start);
    start = Clock::now();
    for (int32_t token : prompt) {
        const auto tick = Clock::now();
        r.trace.push_back(engine.decode(token));
        r.prefill_latencies.push_back(ms(tick));
    }
    r.prefill_ms = ms(start);
    start = Clock::now();
    for (int i = 0; i < steps; ++i) {
        const auto tick = Clock::now();
        const auto & logits = r.trace.back().logits;
        const int32_t greedy = int32_t(std::max_element(logits.begin(), logits.end()) - logits.begin());
        const int32_t token = replay ? replay->at(size_t(i)) : greedy;
        r.replay_greedy_matches &= token == greedy;
        r.generated.push_back(token);
        r.trace.push_back(engine.decode(token));
        r.decode_latencies.push_back(ms(tick));
    }
    r.decode_ms = steps ? ms(start) : 0;
    r.stats = engine.statistics();
    // Fingerprinting and reporting are outside the timed inference regions.
    uint64_t hash = hash_basis;
    for (const auto & step : r.trace) {
        hash_bytes(hash, &step.position, sizeof(step.position));
        hash_bytes(hash, step.logits.data(), step.logits.size() * sizeof(float));
        for (const auto & selection : step.selections) {
            hash_bytes(hash, &selection.layer, sizeof(selection.layer));
            hash_bytes(hash, selection.tokens.data(), selection.tokens.size() * sizeof(int32_t));
        }
    }
    r.checksum = hex(hash);
    return r;
}
void metric(std::ostream & out, const std::vector<double> & times, int tokens,
            const std::vector<double> & latencies) {
    double sum = 0; std::vector<double> rates;
    for (double t : times) { sum += t; if (tokens && t > 0) rates.push_back(1000.0 * tokens / t); }
    out << "{\"tokens_per_run\":" << tokens << ",\"tokens_per_second\":";
    if (tokens && sum > 0) out << 1000.0 * tokens * times.size() / sum; else out << "null";
    out << ",\"run_ms\":"; report::distribution(out, times);
    out << ",\"run_tokens_per_second\":"; report::distribution(out, rates);
    out << ",\"token_latency_ms\":"; report::distribution(out, latencies); out << '}';
}
void write_file(const std::string & path, const std::string & contents) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot open output: " + path);
    out << contents; out.close();
    if (!out) throw std::runtime_error("Cannot write output: " + path);
}
}

int main(int argc, char ** argv) {
    try {
        Args args(argc, argv, {"--help", "--model", "--backend", "--device", "--threads", "--prompt", "--tokens",
                              "--steps", "--warmup", "--repetitions", "--profile", "--profile-limit", "--nvtx", "--json", "--trace", "--label"});
        if (!args.get("--help").empty()) {
            std::cout << "tidal-bench --model MODEL [--backend cuda|cpu] [--prompt 24 | --tokens 1,2,3]\n"
                         "  [--steps 32] [--warmup 1] [--repetitions 5] [--threads 4] [--device 0]\n"
                         "  [--profile off|phases|nodes] [--profile-limit 500000] [--nvtx 0|1] [--json result.json] [--trace trace.json] [--label NAME]\n"
                         "Throughput excludes model load/reset/warmup. Profiling uses a separate replay.\n"
                         "Nodes mode disables fusion and synchronizes each ggml op; timings are host wall time, not CUDA event time.\n";
            return 0;
        }
        const auto path = args.get("--model"), mode_name = args.get("--profile", "off");
        if (path.empty()) throw std::invalid_argument("--model is required");
        tidal::ProfileMode mode;
        if (mode_name == "off") mode = tidal::ProfileMode::Off;
        else if (mode_name == "phases") mode = tidal::ProfileMode::Phases;
        else if (mode_name == "nodes") mode = tidal::ProfileMode::Nodes;
        else throw std::invalid_argument("--profile must be off, phases or nodes");
        const int steps = args.integer("--steps", 32), warmup = args.integer("--warmup", 1);
        const int repetitions = args.integer("--repetitions", 5), nvtx = args.integer("--nvtx", 0);
        if (steps < 0 || warmup < 0 || warmup > 100 || repetitions < 1 || repetitions > 1000 || (nvtx != 0 && nvtx != 1))
            throw std::invalid_argument("Invalid steps/warmup/repetitions/nvtx");
        const auto json_path = args.get("--json"), trace_path = args.get("--trace");
        if (!trace_path.empty() && mode == tidal::ProfileMode::Off) throw std::invalid_argument("--trace requires --profile phases or nodes");
        const auto normalized = [](const std::string & p) { return std::filesystem::weakly_canonical(p); };
        if ((!json_path.empty() && normalized(json_path) == normalized(path)) ||
            (!trace_path.empty() && normalized(trace_path) == normalized(path)) ||
            (!json_path.empty() && !trace_path.empty() && normalized(json_path) == normalized(trace_path)))
            throw std::invalid_argument("Model, JSON and trace paths must differ");
        tidal::Options options{args.get("--backend", "cuda"), args.integer("--device", 0), args.integer("--threads", 4)};
        options.nvtx = nvtx != 0;
        const int event_limit = args.integer("--profile-limit", 500000);
        if (event_limit < 1) throw std::invalid_argument("--profile-limit must be positive");
        options.profile_event_limit = uint64_t(event_limit);
        const auto fingerprint = file_hash(path);
        const auto load_start = Clock::now();
        tidal::Engine engine(path, options);
        const double load_ms = ms(load_start);
        std::vector<int32_t> prompt;
        if (args.contains("--tokens")) {
            if (args.contains("--prompt")) throw std::invalid_argument("Use either --prompt or --tokens");
            const auto text = args.get("--tokens");
            if (text.empty() || text.back() == ',') throw std::invalid_argument("Empty prompt token");
            std::istringstream input(text);
            for (std::string item; std::getline(input, item, ',');) prompt.push_back(number(item));
        } else {
            const int count = args.integer("--prompt", 24);
            if (count < 1 || count > engine.config().context) throw std::invalid_argument("Invalid prompt length");
            for (int i = 0; i < count; ++i) prompt.push_back(int32_t((int64_t(i) * 17 + 1) % engine.config().vocab));
        }
        if (prompt.size() + size_t(steps) > size_t(engine.config().context)) throw std::invalid_argument("Prompt plus steps exceeds context");
        for (auto token : prompt) if (token < 0 || token >= engine.config().vocab) throw std::invalid_argument("Prompt token outside vocabulary");
        for (int i = 0; i < warmup; ++i) execute(engine, prompt, steps);
        std::vector<Run> runs;
        std::vector<tidal::Step> baseline;
        std::vector<double> prefill_times, decode_times, total_times, prefill_latencies, decode_latencies, total_latencies;
        for (int i = 0; i < repetitions; ++i) {
            auto r = execute(engine, prompt, steps);
            if (!runs.empty() && (r.checksum != runs[0].checksum || r.generated != runs[0].generated))
                throw std::runtime_error("Measured repetitions produced different outputs");
            if (i == 0) baseline = std::move(r.trace);
            r.trace.clear(); r.trace.shrink_to_fit();
            prefill_times.push_back(r.prefill_ms); decode_times.push_back(r.decode_ms);
            total_times.push_back(r.prefill_ms + r.decode_ms);
            prefill_latencies.insert(prefill_latencies.end(), r.prefill_latencies.begin(), r.prefill_latencies.end());
            decode_latencies.insert(decode_latencies.end(), r.decode_latencies.begin(), r.decode_latencies.end());
            runs.push_back(std::move(r));
        }
        total_latencies = prefill_latencies; total_latencies.insert(total_latencies.end(), decode_latencies.begin(), decode_latencies.end());
        Run diagnostic;
        double profile_error = 0, profile_nrmse = 0;
        bool selection_match = true;
        if (mode != tidal::ProfileMode::Off) {
            engine.set_profile(mode);
            diagnostic = execute(engine, prompt, steps, &runs[0].generated);
            for (size_t i = 0; i < baseline.size(); ++i) {
                double squared = 0, energy = 0;
                for (size_t j = 0; j < baseline[i].logits.size(); ++j) {
                    const double expected = baseline[i].logits[j], delta = diagnostic.trace[i].logits[j] - expected;
                    profile_error = std::max(profile_error, std::abs(delta)); squared += delta * delta; energy += expected * expected;
                }
                profile_nrmse = std::max(profile_nrmse, std::sqrt(squared / std::max(energy, 1e-30)));
                const auto & a = baseline[i].selections; const auto & b = diagnostic.trace[i].selections;
                selection_match &= a.size() == b.size();
                for (size_t j = 0; j < std::min(a.size(), b.size()); ++j)
                    selection_match &= a[j].layer == b[j].layer && a[j].tokens == b[j].tokens;
            }
            if (mode == tidal::ProfileMode::Phases && diagnostic.checksum != runs[0].checksum)
                throw std::runtime_error("Phase profiling changed inference output");
        }
        const auto groups = report::aggregate(engine.profile_events());
        int driver_version = 0, runtime_version = 0;
#ifdef TIDAL_BENCH_CUDA
        if (options.backend == "cuda") {
            if (cudaDriverGetVersion(&driver_version) != cudaSuccess || cudaRuntimeGetVersion(&runtime_version) != cudaSuccess)
                throw std::runtime_error("Cannot query CUDA driver/runtime versions");
        }
#endif
        std::ostringstream out; out << std::setprecision(12);
        out << "{\"schema_version\":1,\"kind\":\"tidal-benchmark\",\"label\":"; report::string(out, args.get("--label"));
        out << ",\"environment\":{\"backend\":"; report::string(out, engine.backend_name());
        out << ",\"device\":"; report::string(out, engine.backend_description());
        out << ",\"cpu\":"; report::string(out, cpu_name());
        out << ",\"hardware_threads\":" << std::thread::hardware_concurrency() << ",\"threads\":" << options.threads
            << ",\"device_index\":" << options.device << ",\"nvtx\":" << (options.nvtx ? "true" : "false")
            << ",\"build_revision\":"; report::string(out, TIDAL_BUILD_REVISION);
        out << ",\"build_dirty\":"; out << (TIDAL_BUILD_DIRTY ? "true" : "false");
        out << ",\"cuda_driver_version\":" << driver_version << ",\"cuda_runtime_version\":" << runtime_version;
        out << ",\"build_type\":"; report::string(out, TIDAL_BUILD_TYPE);
        out << ",\"compiler\":"; report::string(out, TIDAL_COMPILER);
        out << ",\"ggml_revision\":"; report::string(out, TIDAL_GGML_REVISION); out << '}'
            << ",\"model\":{\"path\":"; report::string(out, path);
        out << ",\"size_bytes\":" << std::filesystem::file_size(path) << ",\"fnv1a64\":"; report::string(out, fingerprint);
        out << ",\"weights\":"; report::model_info(out, engine.model_info());
        out << ",\"config\":"; report::config(out, engine.config()); out << '}'
            << ",\"workload\":{\"prompt\":"; report::array(out, prompt);
        out << ",\"steps\":" << steps << ",\"warmup\":" << warmup << ",\"repetitions\":" << repetitions << '}'
            << ",\"timing\":{\"clock\":\"steady_clock\",\"unit\":\"milliseconds\",\"includes\":\"decode API, transfers, CPU attention, greedy selection; prefill is sequential\","
               "\"excludes\":\"model load, reset, warmup, checksum, serialization, diagnostic replay\"}"
            << ",\"model_load_ms\":" << load_ms << ",\"summary\":{\"prefill\":";
        metric(out, prefill_times, int(prompt.size()), prefill_latencies);
        out << ",\"decode\":"; metric(out, decode_times, steps, decode_latencies);
        out << ",\"total\":"; metric(out, total_times, int(prompt.size()) + steps, total_latencies); out << "},\"runs\":[";
        for (size_t i = 0; i < runs.size(); ++i) {
            if (i) out << ',';
            const auto & r = runs[i];
            out << "{\"reset_ms\":" << r.reset_ms << ",\"prefill_ms\":" << r.prefill_ms << ",\"decode_ms\":" << r.decode_ms
                << ",\"prefill_latency_ms\":"; report::array(out, r.prefill_latencies);
            out << ",\"decode_latency_ms\":"; report::array(out, r.decode_latencies);
            out << ",\"generated\":"; report::array(out, r.generated);
            out << ",\"output_checksum\":"; report::string(out, r.checksum);
            out << ",\"statistics\":"; report::stats(out, r.stats); out << '}';
        }
        out << "],\"profile\":";
        if (mode == tidal::ProfileMode::Off) out << "null";
        else {
            out << "{\"mode\":"; report::string(out, mode_name);
            out << ",\"event_limit\":" << event_limit << ",\"dropped_events\":" << engine.profile_dropped_events();
            out << ",\"clock\":\"host steady_clock; backend operations synchronized\",\"unit\":\"microseconds\","
                   "\"byte_semantics\":\"transfer payload, node logical output, CPU logical KV traffic; not measured DRAM traffic\","
                   "\"separate_replay\":true,\"fixed_generated_tokens\":true,\"inclusive_overlapping_totals\":true,\"fusion_disabled\":"
                << (mode == tidal::ProfileMode::Nodes ? "true" : "false")
                << ",\"logits_max_abs_vs_baseline\":" << profile_error << ",\"logits_max_nrmse_vs_baseline\":" << profile_nrmse
                << ",\"qsa_matches_baseline\":" << (selection_match ? "true" : "false")
                << ",\"greedy_matches_baseline\":" << (diagnostic.replay_greedy_matches ? "true" : "false")
                << ",\"statistics\":"; report::stats(out, diagnostic.stats);
            out << ",\"aggregates\":"; report::aggregates(out, groups);
            out << ",\"events\":"; report::events(out, engine.profile_events(), false); out << '}';
        }
        out << "}\n";
        if (!json_path.empty()) write_file(json_path, out.str());
        if (!trace_path.empty()) {
            std::ostringstream trace; trace << std::setprecision(12) << "{\"displayTimeUnit\":\"ms\",\"dropped_events\":" << engine.profile_dropped_events() << ",\"traceEvents\":";
            report::events(trace, engine.profile_events(), true); trace << "}\n";
            write_file(trace_path, trace.str());
        }
        const auto sum = [](const std::vector<double> & a) { double v = 0; for (double x : a) v += x; return v; };
        std::cout << std::fixed << std::setprecision(3) << "backend=" << engine.backend_name() << "; device=" << engine.backend_description()
                  << "; warmup=" << warmup << "; repetitions=" << repetitions << "; model_load_ms=" << load_ms << '\n'
                  << "prefill_tokens/s=" << 1000.0 * prompt.size() * repetitions / sum(prefill_times)
                  << "; decode_tokens/s=";
        if (steps) std::cout << 1000.0 * steps * repetitions / sum(decode_times); else std::cout << "n/a";
        std::cout << "; total_tokens/s=" << 1000.0 * (prompt.size() + steps) * repetitions / sum(total_times)
                  << "; decode_p50_ms=" << report::percentile(decode_latencies, .5)
                  << "; decode_p95_ms=" << report::percentile(decode_latencies, .95) << '\n';
        if (mode != tidal::ProfileMode::Off) {
            std::cout << "profile=" << mode_name << "; dropped_events=" << engine.profile_dropped_events() << "; events=" << engine.profile_events().size()
                      << "; logits_max_abs_vs_baseline=" << profile_error << "; qsa_matches=" << selection_match
                      << " (separate replay; inclusive synchronized wall times)\n";
            size_t shown = 0;
            for (const auto & a : groups) {
                if (a.example.category == "token" || a.example.category == "layer" || a.example.category == "graph") continue;
                std::cout << a.total / 1000 << " ms  calls=" << a.durations.size() << "  layer=" << a.example.layer
                          << ' ' << a.example.phase << '/' << a.example.category << '/' << a.example.name << '\n';
                if (++shown == 20) break;
            }
        }
        if (!json_path.empty()) std::cout << "json=" << json_path << '\n';
        if (!trace_path.empty()) std::cout << "trace=" << trace_path << '\n';
    } catch (const std::exception & e) { std::cerr << "tidal-bench: " << e.what() << '\n'; return 1; }
}
