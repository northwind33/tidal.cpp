// tidal-server: an HTTP server over one tidal engine. It speaks the OpenAI
// chat/completions API, the Anthropic Messages API and a subset of the
// llama.cpp server endpoints, and serves a small test page at "/".
#include "app.hpp"
#include "args.hpp"
#include "prefill_options.hpp"
#include "sampling_options.hpp"

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <thread>

using namespace tidal::server;

namespace {
HttpServer * active_server = nullptr;
void on_signal(int) { if (active_server) active_server->stop(); }

std::optional<float> real(const Args & args, const char * name) {
    if (!args.contains(name)) return std::nullopt;
    const auto text = args.get(name);
    float value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) throw std::invalid_argument(std::string("Invalid number for ") + name);
    return value;
}
std::optional<int> whole(const Args & args, const char * name) {
    return args.contains(name) ? std::optional<int>(args.integer(name, 0)) : std::nullopt;
}
void log_line(const std::string & line) {
    std::fprintf(stderr, "%s\n", line.c_str());
    std::fflush(stderr);
}
} // namespace

int main(int argc, char ** argv) {
    int exit_code = 0;
    try {
        Args args(argc, argv, prefill_options({"--help", "--model", "--host", "--port", "--alias", "--api-key", "--backend",
            "--device", "--threads", "--context", "--draft-tokens", "--mtp", "--thinking", "--chat-template", "--n-predict",
            "--webui", "--sampler", "--temp", "--top-k", "--top-p", "--min-p", "--repeat-penalty", "--frequency-penalty",
            "--presence-penalty", "--penalty-last-n", "--seed"}));
        if (!args.get("--help").empty()) {
            std::cout <<
                "tidal-server --model MODEL.gguf [--host 127.0.0.1] [--port 8080] [--alias NAME] [--api-key KEY]\n"
                "  [--backend cuda|cpu] [--device 0] [--threads 4] [--context N]\n"
                "  [--draft-tokens 0] [--mtp mtp|SIDECAR.gguf] [--thinking auto|on|off]\n"
                "  [--chat-template auto|chatml|qwen35|qwen38] [--n-predict -1] [--webui 1]\n"
                "  [--sampler device|cpu] [--temp T] [--top-k K] [--top-p P] [--min-p P] [--repeat-penalty R]\n"
                "  [--frequency-penalty F] [--presence-penalty F] [--penalty-last-n N] [--seed S]\n"
                "Serves OpenAI (/v1/chat/completions, /v1/completions, /v1/models), Anthropic (/v1/messages,\n"
                "/v1/messages/count_tokens) and llama.cpp-style (/completion, /infill, /tokenize, /detokenize,\n"
                "/apply-template, /props, /slots, /metrics, /health) endpoints, and a test page at /.\n"
                "One request runs at a time; others wait. The consumed prompt prefix is reused across requests.\n"
                "--context bounds the sequence (prompt plus reply); it defaults to the model's capacity.\n"
                "Sampling flags set the defaults of requests that do not give a value; unset flags use the\n"
                "model's GGUF sampling metadata, then temperature 0.8, top-k 40, top-p 0.95, min-p 0.05.\n"
                "--seed fixes the default seed; otherwise each request without a seed draws a new one.\n"
                "--draft-tokens N generates speculatively with MTP drafts (--mtp, as in tidal-cli).\n"
                "--thinking sets whether chat replies think when a request does not say; auto follows the\n"
                "model's template. --api-key requires 'Authorization: Bearer KEY' or 'x-api-key: KEY'.\n";
            prefill_help(std::cout);
            return 0;
        }
        RuntimeConfig runtime_config;
        runtime_config.model = args.get("--model");
        if (runtime_config.model.empty()) throw std::invalid_argument("--model is required");
        auto & options = runtime_config.options;
        options.backend = args.get("--backend", "cuda");
        options.device = args.integer("--device", 0);
        options.threads = args.integer("--threads", 4);
        options.context = args.integer("--context", 0);
        const auto sampler = args.get("--sampler", "device");
        if (sampler == "cpu") options.sampling.backend = tidal::SamplingBackend::CPU;
        else if (sampler != "device") throw std::invalid_argument("--sampler must be device or cpu");
        configure_prefill(args, options);
        runtime_config.draft_tokens = args.integer("--draft-tokens", 0);
        if (runtime_config.draft_tokens) {
            const auto mtp = args.get("--mtp");
            if (mtp.empty()) throw std::invalid_argument("--draft-tokens requires --mtp");
            if (mtp != "mtp") options.mtp_model = mtp;
        } else if (!args.get("--mtp").empty()) throw std::invalid_argument("--mtp requires --draft-tokens");
        runtime_config.chat_template = args.get("--chat-template", "auto");
        ChatFormat("", runtime_config.chat_template); // Rejects an unknown name before the model loads.

        AppConfig app_config;
        app_config.alias = args.get("--alias");
        app_config.api_key = args.get("--api-key");
        app_config.max_tokens = args.integer("--n-predict", -1);
        const auto webui = args.integer("--webui", 1);
        if (webui != 0 && webui != 1) throw std::invalid_argument("--webui must be 0 or 1");
        app_config.webui = webui != 0;
        const auto thinking = args.get("--thinking", "auto");
        if (thinking == "on") app_config.thinking = true;
        else if (thinking == "off") app_config.thinking = false;
        else if (thinking != "auto") throw std::invalid_argument("--thinking must be auto, on or off");
        auto & s = app_config.sampling;
        s.temperature = real(args, "--temp"); s.top_p = real(args, "--top-p"); s.min_p = real(args, "--min-p");
        s.repeat_penalty = real(args, "--repeat-penalty"); s.frequency_penalty = real(args, "--frequency-penalty");
        s.presence_penalty = real(args, "--presence-penalty");
        s.top_k = whole(args, "--top-k"); s.penalty_last_n = whole(args, "--penalty-last-n");
        if (args.contains("--seed")) {
            const auto text = args.get("--seed");
            uint64_t seed = 0;
            const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), seed);
            if (error != std::errc{} || end != text.data() + text.size()) throw std::invalid_argument("Invalid unsigned 64-bit seed");
            s.seed = seed;
        }
        {
            // Reject bad defaults now rather than on the first request.
            tidal::SamplingConfig check;
            if (s.temperature) check.temperature = *s.temperature;
            if (s.top_p) check.top_p = *s.top_p;
            if (s.min_p) check.min_p = *s.min_p;
            if (s.repeat_penalty) check.repeat_penalty = *s.repeat_penalty;
            if (s.frequency_penalty) check.frequency_penalty = *s.frequency_penalty;
            if (s.presence_penalty) check.presence_penalty = *s.presence_penalty;
            if (s.top_k) check.top_k = *s.top_k;
            if (s.penalty_last_n) check.penalty_last_n = *s.penalty_last_n;
            check.validate();
        }

        // The port opens first so that /health answers while the model loads.
        static HttpServer server;
        const auto host = args.get("--host", "127.0.0.1");
        server.listen(host, args.integer("--port", 8080));
        static Runtime runtime(runtime_config);
        static App app(app_config, runtime, server);
        server.after = [](const Request & request, const Response & response, double ms) {
            char line[512];
            std::snprintf(line, sizeof line, "%s %s %d %.1f ms", request.method.c_str(), request.path.c_str(), response.status(), ms);
            log_line(line);
        };
        active_server = &server;
        std::signal(SIGPIPE, SIG_IGN);
        std::signal(SIGINT, on_signal);
        std::signal(SIGTERM, on_signal);
        log_line("tidal-server: listening on http://" + (host.find(':') == std::string::npos ? host : "[" + host + "]") + ":" +
                 std::to_string(server.port()) + ", loading " + runtime_config.model);
        std::thread([&] {
            while (runtime.state() == Runtime::State::Loading) std::this_thread::sleep_for(std::chrono::milliseconds(20));
            if (runtime.state() == Runtime::State::Failed) {
                log_line("tidal-server: " + runtime.error());
                exit_code = 1;
                server.stop();
                return;
            }
            log_line("tidal-server: model ready (backend " + runtime.backend() + ", context " + std::to_string(runtime.context()) +
                     ", chat format " + runtime.chat().name() + (runtime.tokenizer().byte_fallback() ? ", byte tokenizer" : "") + ")");
        }).detach();
        server.run();
        log_line("tidal-server: shutting down");
        runtime.shutdown();
        runtime.join();
    } catch (const std::exception & e) {
        std::cerr << "tidal-server: " << e.what() << '\n';
        return 1;
    }
    // Connection threads may still be unwinding; skip static destruction.
    std::fflush(nullptr);
    std::_Exit(exit_code);
}
