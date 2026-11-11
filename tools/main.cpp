#include "tidal/tidal.hpp"
#include "args.hpp"
#include "prefill_options.hpp"
#include "sampling_options.hpp"
#include "report.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

template<class T> void array(std::ostream & out, const std::vector<T> & values) {
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) { if (i) out << ','; out << values[i]; }
    out << ']';
}
int main(int argc, char ** argv) {
    try {
        Args args(argc, argv, prefill_options(sampling_options({"--help", "--model", "--backend", "--device", "--threads", "--tokens", "--steps", "--json", "--repeat", "--context", "--stop-at-eos", "--draft-tokens", "--mtp"})));
        if (!args.get("--help").empty()) {
            std::cout << "tidal-cli --model tiny.gguf [--backend cuda|cpu] [--device 0] [--threads 4] "
                         "[--tokens 1,2,3] [--steps 8] [--json trace.json] [--repeat 1] [--context N] [--stop-at-eos 0|1]\n"
                         "  [--draft-tokens 0] [--mtp mtp|SIDECAR.gguf]\n"
                         "--draft-tokens N generates speculatively: MTP drafts (--mtp) are verified in one\n"
                         "CPU-KV hybrid batch per step; it needs --logits 0. --mtp mtp uses the model's own\n"
                         "MTP block, otherwise a sidecar GGUF holding only the MTP tensors.\n";
            sampling_help(std::cout);
            prefill_help(std::cout);
            return 0;
        }
        auto path = args.get("--model");
        if (path.empty()) throw std::invalid_argument("--model is required");
        auto token_text = args.get("--tokens", "1");
        if (token_text.empty() || token_text.back() == ',') throw std::invalid_argument("Empty prompt token");
        std::vector<int32_t> prompt;
        std::istringstream input(token_text);
        for (std::string token; std::getline(input, token, ',');) prompt.push_back(number(token));
        const int steps = args.integer("--steps", 8), repeat = args.integer("--repeat", 1);
        if (steps < 0 || repeat < 1 || repeat > 1000) throw std::invalid_argument("Invalid steps or repeat count");
        tidal::Options options{args.get("--backend", "cuda"), args.integer("--device", 0), args.integer("--threads", 4)};
        options.sampling = sampling_config(args);
        configure_prefill(args, options);
        const bool capture = capture_logits(args, args.contains("--json"));
        options.context = args.integer("--context", 0);
        const int draft_tokens = args.integer("--draft-tokens", 0);
        if (draft_tokens) {
            const auto mtp = args.get("--mtp");
            if (mtp.empty()) throw std::invalid_argument("--draft-tokens requires --mtp");
            if (capture) throw std::invalid_argument("Speculative generation does not capture per-step logits; pass --logits 0");
            options.max_draft_tokens = draft_tokens; options.mtp = true;
            if (mtp != "mtp") options.mtp_model = mtp;
        } else if (!args.get("--mtp").empty()) throw std::invalid_argument("--mtp requires --draft-tokens");
        tidal::Engine engine(path, options);
        const int stop_at_eos = args.integer("--stop-at-eos", engine.config().eos_token >= 0 ? 1 : 0);
        if (stop_at_eos != 0 && stop_at_eos != 1) throw std::invalid_argument("--stop-at-eos must be 0 or 1");
        if (prompt.size() + size_t(steps) > size_t(engine.config().context))
            throw std::invalid_argument("Prompt and generated tokens exceed context capacity");
        std::vector<tidal::Step> trace, initial_trace;
        std::vector<int32_t> generated, initial_generated;
        auto start = std::chrono::steady_clock::now();
        for (int run = 0; run < repeat; ++run) {
            engine.reset(); generated.clear();
            trace = engine.prefill(prompt, capture);
            if (draft_tokens) {
                // The last emitted token is pending; it is consumed by the next verification.
                int32_t next = engine.sample();
                generated.push_back(next);
                while (int(generated.size()) < steps && !(stop_at_eos && next == engine.config().eos_token)) {
                    for (auto token : engine.verify(next, engine.draft(next, draft_tokens))) {
                        if (int(generated.size()) == steps) break;
                        generated.push_back(next = token);
                        if (stop_at_eos && token == engine.config().eos_token) break;
                    }
                }
                if (!steps) generated.clear();
            } else
            for (int i = 0; i < steps; ++i) {
                auto next = engine.sample();
                generated.push_back(next);
                trace.push_back(engine.decode(next, capture));
                if (stop_at_eos && next == engine.config().eos_token) break;
            }
            if (run == 0 && repeat > 1) { initial_trace = trace; initial_generated = generated; }
            if (run > 0) {
                if (generated != initial_generated) throw std::runtime_error("Reset produced different sampled tokens");
                if (trace.size() != initial_trace.size())
                    throw std::runtime_error("Reset/replay produced a different trace length");
                for (size_t i = 0; i < trace.size(); ++i) {
                    const auto & step = trace[i];
                    const auto & first = initial_trace[i];
                    if (step.position != first.position || step.logits != first.logits)
                        throw std::runtime_error("Reset/replay produced different positions or logits");
                    if (step.selections.size() != first.selections.size())
                        throw std::runtime_error("Reset/replay produced different attention layer counts");
                    for (size_t j = 0; j < step.selections.size(); ++j)
                        if (step.selections[j].layer != first.selections[j].layer ||
                            step.selections[j].tokens != first.selections[j].tokens)
                            throw std::runtime_error("Reset/replay produced different attention selections");
                }
            }
        }
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        const auto & s = engine.statistics();
        std::cout << "model="; report::model_info(std::cout, engine.model_info()); std::cout << '\n';
        const bool full = engine.config().architecture == tidal::Architecture::Qwen35;
        std::cout << "backend=" << engine.backend_name() << (full ? "; full_attention=" : "; sparse_attention=")
                  << decode_attention_backend(options) << "; KV=" << kv_type_name(options)
                  << "; prefill_attention=" << (options.backend == "cuda" ? "CUDA" : "CPU") << "; tokens=" << trace.size() << "; repeats=" << repeat << '\n';
        std::cout << "sampler=" << engine.sampling_backend_name() << "; sampling_ms=" << s.sampling_ms << '\n';
        std::cout << "generated="; array(std::cout, generated); std::cout << '\n';
        std::cout << "wall_seconds=" << seconds << "; graph_ms=" << s.graph_ms << "; cpu_attention_ms=" << s.attention_ms
                  << "; transfer_ms=" << s.transfer_ms << "; D2H_bytes=" << s.device_to_host_bytes
                  << "; H2D_bytes=" << s.host_to_device_bytes << '\n';
        const auto json_path = args.get("--json");
        if (!json_path.empty()) {
            std::ofstream out(json_path);
            if (!out) throw std::runtime_error("Cannot open JSON trace: " + json_path);
            out << std::setprecision(9) << "{\"backend\":\"" << engine.backend_name()
                << "\",\"kv_type\":\"" << kv_type_name(options)
                << "\",\"kv_placement\":\"" << (options.kv_placement == tidal::KVPlacement::GPU ? "gpu" : "cpu") << "\",\"generated\":";
            array(out, generated);
            if (!full) out << ",\"sparse_attention_backend\":\"" << decode_attention_backend(options) << '"';
            out << ",\"architecture\":\"" << (full ? "qwen35" : "tidal-qwen4exp")
                << "\",\"attention_kind\":\"" << (full ? "full" : "sparse") << "\",\"attention_backend\":\"" << decode_attention_backend(options) << '"';
            out << ",\"prefill\":"; prefill_json(out, options, prompt.size());
            out << ",\"sampling\":"; sampling_json(out, options.sampling);
            out << ",\"sampling_backend\":"; report::string(out, engine.sampling_backend_name());
            out << ",\"capture_logits\":" << (capture ? "true" : "false");
            out << ",\"model\":"; report::model_info(out, engine.model_info());
            out << ",\"statistics\":{\"backend_graph_nodes\":" << s.device_graph_nodes
                << ",\"sparse_attention_calls\":" << s.sparse_attention_calls
                << ",\"full_attention_calls\":" << s.full_attention_calls
                << ",\"device_to_host_bytes\":" << s.device_to_host_bytes
                << ",\"host_to_device_bytes\":" << s.host_to_device_bytes
                << ",\"graph_ms\":" << s.graph_ms << ",\"attention_ms\":" << s.attention_ms
                << ",\"transfer_ms\":" << s.transfer_ms;
            sampling_stats(out, s); prefill_stats(out, s); out << "},\"trace\":[";
            for (size_t i = 0; i < trace.size(); ++i) {
                if (i) out << ',';
                out << "{\"position\":" << trace[i].position << ",\"logits\":";
                array(out, trace[i].logits);
                out << ",\"selections\":[";
                for (size_t j = 0; j < trace[i].selections.size(); ++j) {
                    if (j) out << ',';
                    out << "{\"layer\":" << trace[i].selections[j].layer << ",\"tokens\":";
                    array(out, trace[i].selections[j].tokens); out << '}';
                }
                out << "]}";
            }
            out << "]}\n";
            out.close();
            if (!out) throw std::runtime_error("Failed to write JSON trace");
        }
        return 0;
    } catch (const std::exception & e) { std::cerr << "tidal-cli: " << e.what() << '\n'; return 1; }
}
