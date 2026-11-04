#include "tidal/tidal.hpp"
#include "args.hpp"

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
        Args args(argc, argv, {"--help", "--model", "--backend", "--device", "--threads", "--tokens", "--steps", "--json", "--repeat"});
        if (!args.get("--help").empty()) {
            std::cout << "tidal-cli --model tiny.gguf [--backend cuda|cpu] [--device 0] [--threads 4] "
                         "[--tokens 1,2,3] [--steps 8] [--json trace.json] [--repeat 1]\n";
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
        tidal::Engine engine(path, options);
        if (prompt.size() + size_t(steps) > size_t(engine.config().context))
            throw std::invalid_argument("Prompt and generated tokens exceed context capacity");
        std::vector<tidal::Step> trace, initial_trace;
        std::vector<int32_t> generated;
        auto start = std::chrono::steady_clock::now();
        for (int run = 0; run < repeat; ++run) {
            engine.reset(); generated.clear();
            trace = engine.prefill(prompt);
            for (int i = 0; i < steps; ++i) {
                const auto & logits = trace.back().logits;
                auto next = int32_t(std::max_element(logits.begin(), logits.end()) - logits.begin());
                generated.push_back(next);
                trace.push_back(engine.decode(next));
            }
            if (run == 0 && repeat > 1) initial_trace = trace;
            if (run > 0) {
                if (trace.size() != initial_trace.size())
                    throw std::runtime_error("Reset/replay produced a different trace length");
                for (size_t i = 0; i < trace.size(); ++i) {
                    const auto & step = trace[i];
                    const auto & first = initial_trace[i];
                    if (step.position != first.position || step.logits != first.logits)
                        throw std::runtime_error("Reset/replay produced different positions or logits");
                    if (step.selections.size() != first.selections.size())
                        throw std::runtime_error("Reset/replay produced different QSA layer counts");
                    for (size_t j = 0; j < step.selections.size(); ++j)
                        if (step.selections[j].layer != first.selections[j].layer ||
                            step.selections[j].tokens != first.selections[j].tokens)
                            throw std::runtime_error("Reset/replay produced different QSA selections");
                }
            }
        }
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        const auto & s = engine.statistics();
        std::cout << "backend=" << engine.backend_name() << "; sparse_attention=CPU; KV=BF16; "
                  << "tokens=" << trace.size() << "; repeats=" << repeat << '\n';
        std::cout << "generated="; array(std::cout, generated); std::cout << '\n';
        std::cout << "wall_seconds=" << seconds << "; graph_ms=" << s.graph_ms << "; cpu_attention_ms=" << s.attention_ms
                  << "; transfer_ms=" << s.transfer_ms << "; D2H_bytes=" << s.device_to_host_bytes
                  << "; H2D_bytes=" << s.host_to_device_bytes << '\n';
        const auto json_path = args.get("--json");
        if (!json_path.empty()) {
            std::ofstream out(json_path);
            if (!out) throw std::runtime_error("Cannot open JSON trace: " + json_path);
            out << std::setprecision(9) << "{\"backend\":\"" << engine.backend_name()
                << "\",\"sparse_attention_backend\":\"CPU\",\"kv_type\":\"BF16\",\"generated\":";
            array(out, generated);
            out << ",\"statistics\":{\"backend_graph_nodes\":" << s.device_graph_nodes
                << ",\"sparse_attention_calls\":" << s.sparse_attention_calls
                << ",\"device_to_host_bytes\":" << s.device_to_host_bytes
                << ",\"host_to_device_bytes\":" << s.host_to_device_bytes
                << ",\"graph_ms\":" << s.graph_ms << ",\"attention_ms\":" << s.attention_ms
                << ",\"transfer_ms\":" << s.transfer_ms << "},\"trace\":[";
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
