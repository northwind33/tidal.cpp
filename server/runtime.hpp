#pragma once
// The model runtime behind the HTTP handlers: one engine with one sequence,
// used from a single worker thread. Requests queue and run one at a time.
#include "chat.hpp"
#include "tokenizer.hpp"
#include "tidal/tidal.hpp"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <thread>

namespace tidal::server {

struct ApiError : std::runtime_error {
    int status;
    std::string type;
    ApiError(int status, std::string type, const std::string & message)
        : std::runtime_error(message), status(status), type(std::move(type)) {}
};
inline ApiError invalid_request(const std::string & message) { return ApiError(400, "invalid_request_error", message); }

struct TokenProbability { int32_t id; float logprob; };
struct GeneratedToken {
    int32_t id;
    float logprob = 0;
    std::vector<TokenProbability> top; // Most likely first; empty unless requested.
};

struct GenerationParams {
    SamplingConfig sampling;
    int max_tokens = -1;            // Negative: until the context is full.
    std::vector<std::string> stop;
    bool cache_prompt = true;       // Reuse the consumed prefix of earlier requests.
    bool logprobs = false;
    int top_logprobs = 0;
    bool render_special = false;    // Include control tokens in the text.
};

enum class StopType { None, Eos, Limit, Word, Cancelled };

struct GenerationResult {
    std::string text;
    std::vector<GeneratedToken> tokens;
    StopType stop = StopType::None;
    std::string stopping_word;
    int prompt_tokens = 0, cached_tokens = 0;
    double prompt_ms = 0, generation_ms = 0;
};

struct GenerationCallbacks {
    // The request reached the engine; nothing has been evaluated yet.
    std::function<void()> on_start;
    // The prompt is evaluated: its token count and how many were reused.
    std::function<void(int, int)> on_prompt;
    // Newly released text with the tokens generated since the last call.
    // Returning false cancels the generation.
    std::function<bool(std::string_view, const std::vector<GeneratedToken> &)> on_text;
    // Polled between tokens; returning true cancels.
    std::function<bool()> cancelled;
};

struct RuntimeConfig {
    std::string model;
    Options options;
    int draft_tokens = 0;       // Speculative decoding with MTP drafts.
    std::string chat_template;  // auto | chatml | qwen35 | qwen38
};

struct Metrics {
    uint64_t requests = 0, prompt_tokens = 0, cached_tokens = 0, generated_tokens = 0;
    double prompt_seconds = 0, generation_seconds = 0;
};

class Runtime {
public:
    // Starts loading the model on the worker thread and returns immediately.
    explicit Runtime(RuntimeConfig config);
    ~Runtime();
    enum class State { Loading, Ready, Failed };
    State state() const { return state_; }
    std::string error() const;

    // Valid once state() is Ready.
    const Tokenizer & tokenizer() const { return *tokenizer_; }
    const ChatFormat & chat() const { return *chat_; }
    const Config & model() const { return model_; }
    const ModelInfo & model_info() const { return model_info_; }
    int context() const { return model_.context; }
    const std::string & backend() const { return backend_; }
    const RuntimeConfig & config() const { return config_; }

    // Queues the request and blocks until it has finished. Callbacks run on the
    // worker thread. Throws ApiError for requests the engine cannot serve.
    GenerationResult generate(const std::vector<int32_t> & prompt, const GenerationParams & params,
                              const GenerationCallbacks & callbacks);
    // Drops the cached sequence and its snapshots.
    void erase();
    // Asks a running generation to stop; later requests are rejected.
    void shutdown();
    // Waits for the worker to finish after shutdown(); the engine is released.
    void join();

    Metrics metrics() const;
    bool busy() const { return busy_; }
    int queued() const { return queued_; }
    int decoded() const { return decoded_; }
    int remaining() const { return remaining_; }

private:
    RuntimeConfig config_;
    std::unique_ptr<Engine> engine_;
    std::unique_ptr<Tokenizer> tokenizer_;
    std::unique_ptr<ChatFormat> chat_;
    Config model_;
    ModelInfo model_info_;
    std::string backend_, error_;
    std::atomic<State> state_{State::Loading};
    std::atomic<bool> busy_{false}, stopping_{false};
    std::atomic<int> queued_{0}, decoded_{0}, remaining_{0};
    mutable std::mutex mutex_;
    std::condition_variable wake_;
    std::deque<std::packaged_task<void()>> jobs_;
    Metrics metrics_;
    std::thread worker_;

    void work();
    void submit(std::function<void()> job);
    GenerationResult execute(const std::vector<int32_t> & prompt, const GenerationParams & params,
                             const GenerationCallbacks & callbacks);
};

} // namespace tidal::server
