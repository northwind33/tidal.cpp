#pragma once
// HTTP API of tidal-server: OpenAI-compatible, Anthropic-compatible and
// llama.cpp-style endpoints over one Runtime.
#include "http.hpp"
#include "runtime.hpp"

#include <optional>

namespace tidal::server {

// Sampling defaults given on the command line; unset fields fall back to the
// model's GGUF metadata and then to the built-in defaults.
struct SamplingOverrides {
    std::optional<float> temperature, top_p, min_p, repeat_penalty, frequency_penalty, presence_penalty;
    std::optional<int> top_k, penalty_last_n;
    std::optional<uint64_t> seed;
};

struct AppConfig {
    std::string alias, api_key;
    SamplingOverrides sampling;
    int max_tokens = -1;            // Default generation limit; negative means until the context is full.
    std::optional<bool> thinking;   // Default for chat requests that do not say; unset follows the template.
    bool webui = true;
};

class App {
public:
    App(AppConfig config, Runtime & runtime, HttpServer & server);
private:
    struct ChatTask;
    struct ChatRun;
    enum class Style { OpenAI, Anthropic };

    AppConfig config_;
    Runtime & runtime_;
    int64_t started_;

    std::string model_name() const;
    SamplingConfig default_sampling() const;
    Json sampling_json(const GenerationParams & params) const;
    void sampling(const Json & body, GenerationParams & params) const;
    std::vector<int32_t> prompt_tokens(const Json & prompt) const;
    void prepare(ChatTask & task) const;
    ChatTask openai_task(const Json & body) const;
    ChatTask anthropic_task(const Json & body, bool counting) const;
    ChatRun run_chat(ChatTask & task, const char * id_prefix, Response & response, std::function<void()> on_start,
                     std::function<void(int, int)> on_prompt,
                     std::function<bool(const ChatDelta &, const std::vector<GeneratedToken> &)> on_delta);
    Json token_logprob(const GeneratedToken & token) const;

    void chat_completions(const Request &, Response &);
    void completions(const Request &, Response &);
    void messages(const Request &, Response &);
    void count_tokens(const Request &, Response &);
    void native_completion(const Request &, Response &, bool infill);
    void completion(const Request & request, Response & response) { native_completion(request, response, false); }
    void infill(const Request & request, Response & response) { native_completion(request, response, true); }
    void tokenize(const Request &, Response &);
    void detokenize(const Request &, Response &);
    void apply_template(const Request &, Response &);
    void models(const Request &, Response &);
    void props(const Request &, Response &);
    void slots(const Request &, Response &);
    void slot_action(const Request &, Response &);
    void metrics(const Request &, Response &);
    void health(const Request &, Response &);
};

} // namespace tidal::server
