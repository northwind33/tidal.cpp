#include "runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>

namespace tidal::server {
namespace {
double elapsed_ms(std::chrono::steady_clock::time_point since) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - since).count();
}
// Length of the longest suffix of `text` that is a proper prefix of `word`.
size_t partial_suffix(std::string_view text, std::string_view word) {
    for (size_t n = std::min(text.size(), word.size() - 1); n > 0; --n)
        if (text.substr(text.size() - n) == word.substr(0, n)) return n;
    return 0;
}
} // namespace

Runtime::Runtime(RuntimeConfig config) : config_(std::move(config)) {
    worker_ = std::thread([this] { work(); });
}

Runtime::~Runtime() {
    shutdown();
    join();
}
void Runtime::join() { if (worker_.joinable()) worker_.join(); }

void Runtime::shutdown() {
    { std::lock_guard lock(mutex_); stopping_ = true; }
    wake_.notify_all();
}

std::string Runtime::error() const { std::lock_guard lock(mutex_); return error_; }
Metrics Runtime::metrics() const { std::lock_guard lock(mutex_); return metrics_; }

void Runtime::work() {
    // The engine is created, used and destroyed on this thread only.
    try {
        auto options = config_.options;
        if (config_.draft_tokens) { options.max_draft_tokens = config_.draft_tokens; options.mtp = true; }
        engine_ = std::make_unique<Engine>(config_.model, options);
        model_ = engine_->config();
        model_info_ = engine_->model_info();
        backend_ = engine_->backend_name();
        tokenizer_ = std::make_unique<Tokenizer>(config_.model, model_.vocab);
        chat_ = std::make_unique<ChatFormat>(tokenizer_->metadata().chat_template, config_.chat_template);
        state_ = State::Ready;
    } catch (const std::exception & e) {
        { std::lock_guard lock(mutex_); error_ = e.what(); }
        engine_.reset();
        state_ = State::Failed;
    }
    for (;;) {
        std::packaged_task<void()> job;
        {
            std::unique_lock lock(mutex_);
            wake_.wait(lock, [&] { return stopping_ || !jobs_.empty(); });
            if (jobs_.empty()) break;
            job = std::move(jobs_.front()); jobs_.pop_front();
        }
        --queued_;
        busy_ = true;
        job();
        busy_ = false;
    }
    engine_.reset();
}

void Runtime::submit(std::function<void()> job) {
    std::packaged_task<void()> task(std::move(job));
    auto done = task.get_future();
    {
        std::lock_guard lock(mutex_);
        if (stopping_) throw ApiError(503, "unavailable_error", "Server is shutting down");
        ++queued_;
        jobs_.push_back(std::move(task));
    }
    wake_.notify_all();
    done.get();
}

void Runtime::erase() {
    if (state_ != State::Ready) throw ApiError(503, "unavailable_error", "Loading model");
    submit([this] { engine_->reset(); });
}

GenerationResult Runtime::generate(const std::vector<int32_t> & prompt, const GenerationParams & params,
                                   const GenerationCallbacks & callbacks) {
    if (state_ != State::Ready) throw ApiError(503, "unavailable_error", "Loading model");
    if (prompt.empty()) throw invalid_request("The prompt is empty");
    if (prompt.size() >= size_t(model_.context))
        throw ApiError(400, "exceed_context_size_error", "The request has " + std::to_string(prompt.size()) +
            " prompt tokens, which leaves no room in the context of " + std::to_string(model_.context) + " tokens");
    try { params.sampling.validate(); }
    catch (const std::invalid_argument &) { throw invalid_request("Invalid sampling parameters"); }
    GenerationResult result;
    submit([&] { result = execute(prompt, params, callbacks); });
    return result;
}

GenerationResult Runtime::execute(const std::vector<int32_t> & prompt, const GenerationParams & params,
                                  const GenerationCallbacks & callbacks) {
    GenerationResult result;
    result.prompt_tokens = int(prompt.size());
    const auto is_cancelled = [&] { return stopping_ || (callbacks.cancelled && callbacks.cancelled()); };
    if (is_cancelled()) { result.stop = StopType::Cancelled; return result; }
    if (callbacks.on_start) callbacks.on_start();
    auto & engine = *engine_;
    int limit = model_.context - int(prompt.size());
    if (params.max_tokens >= 0) limit = std::min(limit, params.max_tokens);
    decoded_ = 0; remaining_ = limit;
    try {
        const auto prompt_start = std::chrono::steady_clock::now();
        if (!params.cache_prompt) engine.reset();
        const uint64_t reused = engine.statistics().prompt_cache_reused_tokens;
        engine.prefill_cached(prompt, false);
        result.cached_tokens = int(engine.statistics().prompt_cache_reused_tokens - reused);
        engine.set_sampling(params.sampling);
        result.prompt_ms = elapsed_ms(prompt_start);
        if (callbacks.on_prompt) callbacks.on_prompt(result.prompt_tokens, result.cached_tokens);

        const auto generation_start = std::chrono::steady_clock::now();
        std::string bytes, pending; // Undecoded token bytes; decoded but unreleased text.
        std::vector<GeneratedToken> fresh;
        bool done = false;
        // Moves complete characters to `pending` and releases what cannot be
        // the beginning of a stop string.
        const auto release = [&](bool final) {
            const size_t complete = final ? bytes.size() : utf8_complete_prefix(bytes);
            pending += sanitize_utf8(std::string_view(bytes).substr(0, complete));
            bytes.erase(0, complete);
            size_t length = pending.size();
            const std::string * word = nullptr;
            for (const auto & stop : params.stop) {
                const auto at = pending.find(stop);
                if (at != std::string::npos && (!word || at < length)) { length = at; word = &stop; }
            }
            if (word) { result.stop = StopType::Word; result.stopping_word = *word; done = true; }
            else if (!final) {
                size_t hold = 0;
                for (const auto & stop : params.stop) hold = std::max(hold, partial_suffix(pending, stop));
                length -= hold;
            }
            if (!length && fresh.empty()) return;
            const std::string_view text = std::string_view(pending).substr(0, length);
            result.text += text;
            if (callbacks.on_text && !callbacks.on_text(text, fresh)) { result.stop = StopType::Cancelled; done = true; }
            fresh.clear();
            pending.erase(0, length);
        };
        // Accounts for one sampled token; false ends the generation.
        const auto accept = [&](GeneratedToken token) {
            if (tokenizer_->is_end(token.id) || token.id == model_.eos_token) { result.stop = StopType::Eos; return false; }
            bytes += tokenizer_->piece(token.id, params.render_special);
            result.tokens.push_back(token); fresh.push_back(std::move(token));
            decoded_ = int(result.tokens.size()); remaining_ = limit - decoded_;
            release(false);
            if (done) return false;
            if (int(result.tokens.size()) >= limit) { result.stop = StopType::Limit; return false; }
            if (result.tokens.size() % 4 == 0 && is_cancelled()) { result.stop = StopType::Cancelled; return false; }
            return true;
        };
        const auto sample = [&] {
            GeneratedToken token{};
            if (!params.logprobs) { token.id = engine.sample(); return token; }
            const auto logits = engine.logits();
            token.id = engine.sample();
            const float peak = *std::max_element(logits.begin(), logits.end());
            double sum = 0;
            for (const float value : logits) sum += std::exp(double(value - peak));
            const double normalizer = double(peak) + std::log(sum);
            token.logprob = float(double(logits[size_t(token.id)]) - normalizer);
            if (params.top_logprobs > 0) {
                std::vector<int32_t> ids(logits.size());
                std::iota(ids.begin(), ids.end(), 0);
                const size_t count = std::min(ids.size(), size_t(params.top_logprobs));
                std::partial_sort(ids.begin(), ids.begin() + long(count), ids.end(), [&](int32_t a, int32_t b) {
                    return logits[size_t(a)] != logits[size_t(b)] ? logits[size_t(a)] > logits[size_t(b)] : a < b;
                });
                for (size_t i = 0; i < count; ++i) token.top.push_back({ids[i], float(double(logits[size_t(ids[i])]) - normalizer)});
            }
            return token;
        };
        if (limit <= 0) result.stop = StopType::Limit;
        else if (config_.draft_tokens && !params.logprobs) {
            // The last accepted token is pending: the next verification consumes it.
            int32_t next = engine.sample();
            bool more = accept({next, 0, {}});
            while (more) {
                const int room = std::min({config_.draft_tokens, limit - int(result.tokens.size()) - 1,
                                           model_.context - int(engine.tokens().size()) - 1});
                if (room < 1) {
                    engine.decode(next, false);
                    more = accept({next = engine.sample(), 0, {}});
                    continue;
                }
                for (const auto token : engine.verify(next, engine.draft(next, room))) {
                    next = token;
                    if (!(more = accept({token, 0, {}}))) break;
                }
            }
        } else {
            for (auto next = sample(); accept(next); next = sample()) engine.decode(next.id, false);
        }
        if (result.stop != StopType::Word && result.stop != StopType::Cancelled) {
            const auto stop = result.stop;
            release(true);
            // Flushing can still complete a stop string or lose the client.
            if (result.stop != StopType::Word && result.stop != StopType::Cancelled) result.stop = stop;
        }
        result.generation_ms = elapsed_ms(generation_start);
    } catch (const ApiError &) {
        throw;
    } catch (const std::exception & e) {
        // The engine refuses further work after a failure until it is reset.
        try { engine.reset(); } catch (...) {}
        throw ApiError(500, "server_error", std::string("Inference failed: ") + e.what());
    }
    decoded_ = 0; remaining_ = 0;
    std::lock_guard lock(mutex_);
    ++metrics_.requests;
    metrics_.prompt_tokens += uint64_t(result.prompt_tokens - result.cached_tokens);
    metrics_.cached_tokens += uint64_t(result.cached_tokens);
    metrics_.generated_tokens += result.tokens.size();
    metrics_.prompt_seconds += result.prompt_ms / 1000;
    metrics_.generation_seconds += result.generation_ms / 1000;
    return result;
}

} // namespace tidal::server
