#include "app.hpp"
#include "webui.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>

namespace tidal::server {

struct App::ChatTask {
    std::string model;
    std::vector<Message> messages;
    ChatOptions options;
    GenerationParams params;
    bool stream = false, include_usage = false, inline_reasoning = false;
    ChatPrompt prompt;
    std::vector<int32_t> tokens;
};
struct App::ChatRun {
    std::string reasoning, content;
    std::vector<ToolCall> calls;
    GenerationResult result;
};

namespace {

const char * const json_type = "application/json; charset=utf-8";
const char * const version = "tidal-server 0.1.0";

bool trim_empty(const std::string & text) { return text.find_first_not_of(" \t\r\n") == std::string::npos; }
Json list(std::initializer_list<Json> items) { return Json(Json::Array(items)); }

uint64_t random_u64() {
    thread_local std::mt19937_64 generator(std::random_device{}() ^
        uint64_t(std::chrono::steady_clock::now().time_since_epoch().count()));
    return generator();
}
std::string random_token(size_t length) {
    static const char alphabet[] = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    std::string out;
    for (size_t i = 0; i < length; ++i) out += alphabet[random_u64() % 62];
    return out;
}
int64_t unix_time() {
    return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

Json openai_error(int status, const std::string & type, const std::string & message) {
    return Json::object({{"error", Json::object({{"code", status}, {"message", message}, {"type", type}})}});
}
Json anthropic_error(int status, const std::string & type, const std::string & message) {
    const char * name = status == 400 || status == 501 ? "invalid_request_error" : status == 401 ? "authentication_error"
        : status == 404 ? "not_found_error" : status == 413 ? "request_too_large" : status == 503 ? "overloaded_error" : "api_error";
    (void) type;
    return Json::object({{"type", "error"}, {"error", Json::object({{"type", name}, {"message", message}})}});
}
void send_json(Response & response, int status, const Json & value) { response.send(status, json_type, value.dump()); }
bool sse(Response & response, const Json & data) { return response.write("data: " + data.dump() + "\n\n"); }
bool sse(Response & response, const char * event, const Json & data) {
    return response.write(std::string("event: ") + event + "\ndata: " + data.dump() + "\n\n");
}

Json parse_body(const Request & request) {
    if (request.body.empty()) return Json::object();
    Json body;
    try { body = Json::parse(request.body); }
    catch (const JsonError & e) { throw invalid_request(e.what()); }
    if (!body.is_object()) throw invalid_request("The request body must be a JSON object");
    return body;
}
// The value of `key` unless absent or null.
const Json * present(const Json & body, std::string_view key) {
    const auto * value = body.find(key);
    return value && !value->is_null() ? value : nullptr;
}
std::optional<double> number(const Json & body, std::string_view key) {
    const auto * value = present(body, key);
    if (!value) return std::nullopt;
    if (!value->is_number()) throw invalid_request("'" + std::string(key) + "' must be a number");
    return value->number();
}
std::optional<int64_t> integer(const Json & body, std::string_view key) {
    const auto value = number(body, key);
    if (!value) return std::nullopt;
    if (std::floor(*value) != *value || std::fabs(*value) > 9e15) throw invalid_request("'" + std::string(key) + "' must be an integer");
    return int64_t(*value);
}
std::optional<bool> boolean(const Json & body, std::string_view key) {
    const auto * value = present(body, key);
    if (!value) return std::nullopt;
    if (!value->is_bool()) throw invalid_request("'" + std::string(key) + "' must be a boolean");
    return value->boolean();
}
std::optional<std::string> text(const Json & body, std::string_view key) {
    const auto * value = present(body, key);
    if (!value) return std::nullopt;
    if (!value->is_string()) throw invalid_request("'" + std::string(key) + "' must be a string");
    return value->string();
}
std::vector<std::string> stop_words(const Json & body, std::string_view key) {
    std::vector<std::string> words;
    const auto * value = present(body, key);
    if (!value) return words;
    if (value->is_string()) { if (!value->string().empty()) words.push_back(value->string()); return words; }
    if (!value->is_array()) throw invalid_request("'" + std::string(key) + "' must be a string or an array of strings");
    for (const auto & item : value->items()) {
        if (!item.is_string()) throw invalid_request("'" + std::string(key) + "' must contain only strings");
        if (!item.string().empty()) words.push_back(item.string());
    }
    return words;
}

// Message content: a string, null, or an array of text parts. Parts are joined
// with `separator`. Other modalities are rejected: the engine is text-only.
std::string content_text(const Json & content, const char * separator = "") {
    if (content.is_null()) return {};
    if (content.is_string()) return content.string();
    if (!content.is_array()) throw invalid_request("Message content must be a string or an array of content parts");
    std::string out;
    bool first = true;
    for (const auto & part : content.items()) {
        if (part.is_string()) { if (!first) out += separator; out += part.string(); first = false; continue; }
        const auto type = part.at("type").is_string() ? part.at("type").string() : std::string();
        if ((type == "text" || type.empty()) && part.at("text").is_string()) {
            if (!first) out += separator;
            out += part.at("text").string(); first = false;
        } else throw invalid_request("Unsupported content part '" + type + "': this server accepts text only");
    }
    return out;
}

Json timings(const GenerationResult & r) {
    const int evaluated = r.prompt_tokens - r.cached_tokens, predicted = int(r.tokens.size());
    const auto rate = [](int count, double ms) { return ms > 0 ? Json(count * 1000.0 / ms) : Json(); };
    const auto each = [](int count, double ms) { return count > 0 ? Json(ms / count) : Json(); };
    return Json::object({{"cache_n", r.cached_tokens}, {"prompt_n", evaluated}, {"prompt_ms", r.prompt_ms},
        {"prompt_per_token_ms", each(evaluated, r.prompt_ms)}, {"prompt_per_second", rate(evaluated, r.prompt_ms)},
        {"predicted_n", predicted}, {"predicted_ms", r.generation_ms},
        {"predicted_per_token_ms", each(predicted, r.generation_ms)}, {"predicted_per_second", rate(predicted, r.generation_ms)}});
}
Json openai_usage(const GenerationResult & r) {
    return Json::object({{"prompt_tokens", r.prompt_tokens}, {"completion_tokens", int(r.tokens.size())},
        {"total_tokens", r.prompt_tokens + int(r.tokens.size())},
        {"prompt_tokens_details", Json::object({{"cached_tokens", r.cached_tokens}})}});
}
Json anthropic_usage(const GenerationResult & r) {
    return Json::object({{"input_tokens", r.prompt_tokens - r.cached_tokens}, {"cache_creation_input_tokens", 0},
        {"cache_read_input_tokens", r.cached_tokens}, {"output_tokens", int(r.tokens.size())}});
}
Json bytes_json(std::string_view bytes) {
    Json out = Json::array();
    for (const char c : bytes) out.push(int(uint8_t(c)));
    return out;
}
Json tool_calls_json(const std::vector<ToolCall> & calls, size_t first_index, bool with_index) {
    Json out = Json::array();
    for (size_t i = 0; i < calls.size(); ++i) {
        Json call = Json::object();
        if (with_index) call["index"] = first_index + i;
        call["id"] = calls[i].id; call["type"] = "function";
        call["function"] = Json::object({{"name", calls[i].name}, {"arguments", calls[i].arguments.dump()}});
        out.push(std::move(call));
    }
    return out;
}

} // namespace

App::App(AppConfig config, Runtime & runtime, HttpServer & server)
    : config_(std::move(config)), runtime_(runtime), started_(unix_time()) {
    const auto is_public = [](const std::string & path) {
        return path == "/health" || path == "/v1/health" || path == "/" || path == "/index.html";
    };
    const auto is_anthropic = [](const std::string & path) { return path.rfind("/v1/messages", 0) == 0; };
    server.before = [=, this](const Request & request, Response & response) {
        if (is_public(request.path)) return true;
        const auto fail = [&](int status, const char * type, const std::string & message) {
            send_json(response, status, is_anthropic(request.path) ? anthropic_error(status, type, message) : openai_error(status, type, message));
            return false;
        };
        if (!config_.api_key.empty() && request.path != "/models" && request.path != "/v1/models") {
            const auto & authorization = request.header("authorization");
            const bool bearer = authorization.rfind("Bearer ", 0) == 0 && authorization.substr(7) == config_.api_key;
            if (!bearer && request.header("x-api-key") != config_.api_key) return fail(401, "authentication_error", "Invalid API key");
        }
        if (runtime_.state() == Runtime::State::Loading) return fail(503, "unavailable_error", "Loading model");
        if (runtime_.state() == Runtime::State::Failed) return fail(500, "server_error", "Model failed to load: " + runtime_.error());
        return true;
    };
    const auto add = [&](const char * method, std::initializer_list<const char *> paths, Style style,
                         void (App::*handler)(const Request &, Response &)) {
        Handler wrapped = [this, style, handler](const Request & request, Response & response) {
            int status = 500;
            std::string type = "server_error", message;
            try { (this->*handler)(request, response); return; }
            catch (const ApiError & e) { status = e.status; type = e.type; message = e.what(); }
            catch (const ChatError & e) { status = 400; type = "invalid_request_error"; message = e.what(); }
            catch (const JsonError & e) { status = 400; type = "invalid_request_error"; message = e.what(); }
            catch (const std::exception & e) { message = e.what(); }
            const Json error = style == Style::Anthropic ? anthropic_error(status, type, message) : openai_error(status, type, message);
            if (!response.started()) send_json(response, status, error);
            else if (response.streaming()) { if (style == Style::Anthropic) sse(response, "error", error); else sse(response, error); }
        };
        for (const char * path : paths) server.route(method, path, wrapped);
    };
    add("GET", {"/health", "/v1/health"}, Style::OpenAI, &App::health);
    add("GET", {"/v1/models", "/models"}, Style::OpenAI, &App::models);
    add("POST", {"/v1/chat/completions", "/chat/completions"}, Style::OpenAI, &App::chat_completions);
    add("POST", {"/v1/completions"}, Style::OpenAI, &App::completions);
    add("POST", {"/v1/messages"}, Style::Anthropic, &App::messages);
    add("POST", {"/v1/messages/count_tokens"}, Style::Anthropic, &App::count_tokens);
    add("POST", {"/tokenize"}, Style::OpenAI, &App::tokenize);
    add("POST", {"/detokenize"}, Style::OpenAI, &App::detokenize);
    add("POST", {"/apply-template"}, Style::OpenAI, &App::apply_template);
    add("GET", {"/props"}, Style::OpenAI, &App::props);
    add("GET", {"/slots"}, Style::OpenAI, &App::slots);
    add("POST", {"/slots/*"}, Style::OpenAI, &App::slot_action);
    add("GET", {"/metrics"}, Style::OpenAI, &App::metrics);
    add("POST", {"/completion", "/completions"}, Style::OpenAI, &App::completion);
    add("POST", {"/infill"}, Style::OpenAI, &App::infill);
    for (const char * path : {"/embedding", "/embeddings", "/v1/embeddings", "/reranking", "/rerank", "/v1/rerank", "/v1/reranking",
                              "/lora-adapters", "/v1/responses", "/v1/audio/transcriptions", "/v1/images/generations"})
        server.route("POST", path, [](const Request &, Response & response) {
            send_json(response, 501, openai_error(501, "not_supported_error", "This endpoint is not supported by tidal-server"));
        });
    if (config_.webui) {
        const Handler page = [](const Request &, Response & response) {
            response.send(200, "text/html; charset=utf-8", std::string_view(reinterpret_cast<const char *>(webui_html), webui_html_size));
        };
        server.route("GET", "/", page);
        server.route("GET", "/index.html", page);
    }
}

std::string App::model_name() const {
    if (!config_.alias.empty()) return config_.alias;
    const auto & path = runtime_.config().model;
    const auto slash = path.find_last_of('/');
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

SamplingConfig App::default_sampling() const {
    SamplingConfig s;
    s.temperature = 0.8f; s.top_k = 40; s.top_p = 0.95f; s.min_p = 0.05f;
    const auto & m = runtime_.tokenizer().metadata();
    const auto & o = config_.sampling;
    const auto pick = [](auto & target, const auto & cli, const auto & model) {
        if (cli) target = *cli; else if (model) target = *model;
    };
    pick(s.temperature, o.temperature, m.temperature); pick(s.top_p, o.top_p, m.top_p);
    pick(s.min_p, o.min_p, m.min_p); pick(s.top_k, o.top_k, m.top_k);
    pick(s.repeat_penalty, o.repeat_penalty, m.repeat_penalty);
    if (o.frequency_penalty) s.frequency_penalty = *o.frequency_penalty;
    if (o.presence_penalty) s.presence_penalty = *o.presence_penalty;
    if (o.penalty_last_n) s.penalty_last_n = *o.penalty_last_n;
    if (o.seed) s.seed = *o.seed;
    return s;
}

void App::sampling(const Json & body, GenerationParams & params) const {
    auto & s = params.sampling;
    s = default_sampling();
    if (const auto v = number(body, "temperature")) s.temperature = float(*v);
    if (const auto v = number(body, "top_p")) s.top_p = float(*v);
    if (const auto v = integer(body, "top_k")) s.top_k = int(std::clamp<int64_t>(*v, -1, 1 << 30));
    if (const auto v = number(body, "min_p")) s.min_p = float(*v);
    if (const auto v = number(body, "repeat_penalty")) s.repeat_penalty = float(*v);
    if (const auto v = integer(body, "repeat_last_n")) s.penalty_last_n = int(std::clamp<int64_t>(*v, -1, 1 << 30));
    if (const auto v = number(body, "presence_penalty")) s.presence_penalty = float(*v);
    if (const auto v = number(body, "frequency_penalty")) s.frequency_penalty = float(*v);
    // top_p 0 asks for the single most likely token.
    if (s.top_p <= 0) { s.temperature = 0; s.top_p = 1; }
    const auto seed = integer(body, "seed");
    if (seed && *seed >= 0) s.seed = uint64_t(*seed);
    else if (!config_.sampling.seed) s.seed = random_u64();
    try { s.validate(); }
    catch (const std::invalid_argument &) {
        throw invalid_request("Invalid sampling parameters: temperature >= 0, 0 <= top_p <= 1, 0 <= min_p <= 1, "
                              "repeat_penalty > 0, top_k >= -1 and repeat_last_n >= -1 are required");
    }
}

Json App::sampling_json(const GenerationParams & params) const {
    const auto & s = params.sampling;
    Json stop = Json::array();
    for (const auto & word : params.stop) stop.push(word);
    return Json::object({{"n_predict", params.max_tokens}, {"seed", s.seed}, {"temperature", s.temperature},
        {"top_k", s.top_k}, {"top_p", s.top_p}, {"min_p", s.min_p}, {"repeat_penalty", s.repeat_penalty},
        {"repeat_last_n", s.penalty_last_n}, {"presence_penalty", s.presence_penalty},
        {"frequency_penalty", s.frequency_penalty}, {"stop", std::move(stop)},
        {"n_probs", params.logprobs ? params.top_logprobs : 0}, {"cache_prompt", params.cache_prompt}});
}

// A prompt is a string, an array of token ids, or an array mixing both.
std::vector<int32_t> App::prompt_tokens(const Json & prompt) const {
    const auto & tokenizer = runtime_.tokenizer();
    std::vector<int32_t> tokens;
    if (prompt.is_string()) tokens = tokenizer.encode(prompt.string(), true);
    else if (prompt.is_array()) {
        for (const auto & item : prompt.items()) {
            if (item.is_string()) {
                const auto part = tokenizer.encode(item.string(), true);
                tokens.insert(tokens.end(), part.begin(), part.end());
            } else if (item.is_int() && item.integer() >= 0 && item.integer() < tokenizer.size()) tokens.push_back(int32_t(item.integer()));
            else throw invalid_request("'prompt' array items must be strings or token ids inside the vocabulary");
        }
    } else throw invalid_request("'prompt' must be a string or an array of strings and token ids");
    if (tokens.empty() && tokenizer.bos() >= 0) tokens.push_back(tokenizer.bos());
    if (tokens.empty()) throw invalid_request("The prompt is empty");
    return tokens;
}

Json App::token_logprob(const GeneratedToken & token) const {
    const auto & tokenizer = runtime_.tokenizer();
    const auto entry = [&](int32_t id, float logprob) {
        const auto piece = tokenizer.piece(id, true);
        return Json::object({{"id", id}, {"token", sanitize_utf8(piece)}, {"bytes", bytes_json(piece)}, {"logprob", logprob}});
    };
    Json out = entry(token.id, token.logprob);
    Json top = Json::array();
    for (const auto & candidate : token.top) top.push(entry(candidate.id, candidate.logprob));
    out["top_logprobs"] = std::move(top);
    return out;
}

void App::prepare(ChatTask & task) const {
    if (!runtime_.chat().supports_tools()) task.options.tools = Json();
    task.prompt = runtime_.chat().render(task.messages, task.options);
    task.tokens = runtime_.tokenizer().encode(task.prompt.text, true);
}

App::ChatTask App::openai_task(const Json & body) const {
    ChatTask task;
    task.model = text(body, "model").value_or(model_name());
    const auto * messages = present(body, "messages");
    if (!messages || !messages->is_array() || !messages->size()) throw invalid_request("'messages' must be a non-empty array");
    bool only_system = true;
    for (const auto & item : messages->items()) {
        if (!item.is_object() || !item.at("role").is_string()) throw invalid_request("Each message needs a string 'role'");
        Message message;
        message.role = item.at("role").string();
        if (message.role == "developer") message.role = "system";
        if (message.role == "function") message.role = "tool";
        if (message.role != "system" && message.role != "user" && message.role != "assistant" && message.role != "tool")
            throw invalid_request("Unexpected message role '" + message.role + "'");
        message.content = content_text(item.at("content"));
        if (message.role == "assistant") {
            if (item.at("reasoning_content").is_string()) message.reasoning = item.at("reasoning_content").string();
            if (const auto * calls = present(item, "tool_calls")) {
                if (!calls->is_array()) throw invalid_request("'tool_calls' must be an array");
                for (const auto & entry : calls->items()) {
                    const auto & function = entry.at("function").is_object() ? entry.at("function") : entry;
                    if (!function.at("name").is_string()) throw invalid_request("A tool call needs a function name");
                    ToolCall call;
                    call.name = function.at("name").string();
                    if (entry.at("id").is_string()) call.id = entry.at("id").string();
                    const auto & arguments = function.at("arguments");
                    if (arguments.is_object()) call.arguments = arguments;
                    else if (arguments.is_string() && !trim_empty(arguments.string())) {
                        try { call.arguments = Json::parse(arguments.string()); }
                        catch (const JsonError &) { throw invalid_request("Tool call arguments must be a JSON object"); }
                        if (!call.arguments.is_object()) throw invalid_request("Tool call arguments must be a JSON object");
                    }
                    message.tool_calls.push_back(std::move(call));
                }
            }
        }
        // Several leading system messages become one; the template allows a single one.
        if (message.role == "system" && only_system && !task.messages.empty()) {
            task.messages.back().content += "\n\n" + message.content;
            continue;
        }
        only_system = only_system && message.role == "system";
        task.messages.push_back(std::move(message));
    }
    if (const auto * tools = present(body, "tools")) {
        if (!tools->is_array()) throw invalid_request("'tools' must be an array");
        for (const auto & tool : tools->items())
            if (!tool.at("function").at("name").is_string()) throw invalid_request("Each tool needs function.name");
        task.options.tools = *tools;
    }
    if (const auto * choice = present(body, "tool_choice")) {
        if (choice->is_string()) {
            if (choice->string() == "none") task.options.tools = Json();
            else if (choice->string() == "required") task.options.forced_tool = "*";
            else if (choice->string() != "auto") throw invalid_request("'tool_choice' must be none, auto, required or a function");
        } else if (choice->at("function").at("name").is_string()) task.options.forced_tool = choice->at("function").at("name").string();
        else throw invalid_request("'tool_choice' must be none, auto, required or a function");
        if (!task.options.forced_tool.empty() && !task.options.tools.size()) throw invalid_request("'tool_choice' requires 'tools'");
    }
    // Thinking: template arguments first, then the common request spellings.
    task.options.enable_thinking = config_.thinking;
    const auto effort = [&](const std::string & value) {
        if (value == "none") { task.options.enable_thinking = false; return; }
        task.options.enable_thinking = true;
        task.options.reasoning_effort = value == "minimal" || value == "low" ? "low" : value == "medium" ? "medium" : "xhigh";
    };
    if (const auto value = text(body, "reasoning_effort")) effort(*value);
    if (const auto * thinking = present(body, "thinking"); thinking && thinking->at("type").is_string())
        task.options.enable_thinking = thinking->at("type").string() != "disabled";
    if (const auto value = boolean(body, "enable_thinking")) task.options.enable_thinking = *value;
    if (const auto * kwargs = present(body, "chat_template_kwargs")) {
        if (!kwargs->is_object()) throw invalid_request("'chat_template_kwargs' must be an object");
        if (const auto value = boolean(*kwargs, "enable_thinking")) task.options.enable_thinking = *value;
        if (const auto value = text(*kwargs, "reasoning_effort")) task.options.reasoning_effort = *value;
        if (const auto value = boolean(*kwargs, "preserve_thinking")) task.options.preserve_thinking = *value;
    }
    if (const auto value = boolean(body, "add_generation_prompt")) task.options.add_generation_prompt = *value;
    if (const auto value = text(body, "reasoning_format")) task.inline_reasoning = *value == "none";

    sampling(body, task.params);
    task.params.stop = stop_words(body, "stop");
    task.params.max_tokens = config_.max_tokens;
    for (const char * key : {"n_predict", "max_tokens", "max_completion_tokens"})
        if (const auto value = integer(body, key)) task.params.max_tokens = int(std::clamp<int64_t>(*value, -1, 1 << 30));
    task.params.cache_prompt = boolean(body, "cache_prompt").value_or(true);
    task.params.logprobs = boolean(body, "logprobs").value_or(false);
    if (const auto value = integer(body, "top_logprobs")) {
        if (*value < 0 || *value > 20) throw invalid_request("'top_logprobs' must be within 0-20");
        task.params.top_logprobs = int(*value);
        if (*value) task.params.logprobs = true;
    }
    task.stream = boolean(body, "stream").value_or(false);
    if (const auto * options = present(body, "stream_options")) task.include_usage = boolean(*options, "include_usage").value_or(false);
    if (const auto n = integer(body, "n"); n && *n != 1) throw invalid_request("Only n = 1 is supported");
    if (const auto * format = present(body, "response_format"); format && !(format->at("type").is_string() && format->at("type").string() == "text"))
        throw invalid_request("Only response_format type 'text' is supported: constrained output is not implemented");
    for (const char * key : {"grammar", "json_schema"})
        if (present(body, key)) throw invalid_request(std::string("'") + key + "' is not supported: constrained output is not implemented");
    return task;
}

App::ChatTask App::anthropic_task(const Json & body, bool counting) const {
    ChatTask task;
    task.model = text(body, "model").value_or(model_name());
    if (const auto * system = present(body, "system")) {
        const auto content = content_text(*system, "\n\n");
        if (!content.empty()) task.messages.push_back({"system", content, std::nullopt, {}});
    }
    const auto * messages = present(body, "messages");
    if (!messages || !messages->is_array() || !messages->size()) throw invalid_request("messages: Field required");
    for (const auto & item : messages->items()) {
        if (!item.is_object() || !item.at("role").is_string()) throw invalid_request("messages: each message needs a role");
        const auto & role = item.at("role").string();
        const auto & content = item.at("content");
        if (role != "user" && role != "assistant") throw invalid_request("messages: role must be user or assistant");
        Message message;
        message.role = role;
        if (content.is_string()) { message.content = content.string(); task.messages.push_back(std::move(message)); continue; }
        if (!content.is_array()) throw invalid_request("messages: content must be a string or an array of blocks");
        bool any = false;
        const auto flush = [&] {
            if (any) task.messages.push_back(std::move(message));
            message = Message(); message.role = role; any = false;
        };
        for (const auto & block : content.items()) {
            const auto type = block.at("type").is_string() ? block.at("type").string() : std::string();
            if (type == "text" && block.at("text").is_string()) {
                if (any && !message.content.empty() && role == "user") message.content += "\n";
                message.content += block.at("text").string(); any = true;
            } else if (type == "thinking" && role == "assistant") {
                if (block.at("thinking").is_string()) message.reasoning = message.reasoning.value_or("") + block.at("thinking").string();
                any = true;
            } else if (type == "redacted_thinking" && role == "assistant") {
                any = true;
            } else if (type == "tool_use" && role == "assistant") {
                if (!block.at("name").is_string()) throw invalid_request("messages: tool_use needs a name");
                ToolCall call;
                call.name = block.at("name").string();
                if (block.at("id").is_string()) call.id = block.at("id").string();
                if (block.at("input").is_object()) call.arguments = block.at("input");
                message.tool_calls.push_back(std::move(call)); any = true;
            } else if (type == "tool_result" && role == "user") {
                flush();
                std::string result = content_text(block.at("content"), "\n");
                task.messages.push_back({"tool", std::move(result), std::nullopt, {}});
            } else throw invalid_request("messages: unsupported content block '" + type + "' (this server accepts text, thinking and tool blocks)");
        }
        flush();
    }
    if (const auto * tools = present(body, "tools")) {
        if (!tools->is_array()) throw invalid_request("tools: must be an array");
        Json converted = Json::array();
        for (const auto & tool : tools->items()) {
            // Only client tools carry a schema; server-side tools cannot run here.
            if (!tool.at("name").is_string() || !tool.contains("input_schema")) continue;
            Json function = Json::object({{"name", tool.at("name")}});
            if (tool.at("description").is_string()) function["description"] = tool.at("description");
            function["parameters"] = tool.at("input_schema");
            converted.push(Json::object({{"type", "function"}, {"function", std::move(function)}}));
        }
        if (converted.size()) task.options.tools = std::move(converted);
    }
    if (const auto * choice = present(body, "tool_choice")) {
        const auto type = choice->at("type").is_string() ? choice->at("type").string() : std::string();
        if (type == "none") task.options.tools = Json();
        else if (type == "any") task.options.forced_tool = "*";
        else if (type == "tool" && choice->at("name").is_string()) task.options.forced_tool = choice->at("name").string();
        else if (type != "auto") throw invalid_request("tool_choice: type must be auto, any, tool or none");
        if (!task.options.forced_tool.empty() && !task.options.tools.size()) task.options.forced_tool.clear();
    }
    // Without a `thinking` field the Messages API does not think.
    task.options.enable_thinking = false;
    if (const auto * thinking = present(body, "thinking"); thinking && thinking->at("type").is_string())
        task.options.enable_thinking = thinking->at("type").string() != "disabled";
    if (const auto * output = present(body, "output_config"))
        if (const auto value = text(*output, "effort"))
            task.options.reasoning_effort = *value == "low" ? "low" : *value == "medium" ? "medium" : "xhigh";

    sampling(body, task.params);
    task.params.stop = stop_words(body, "stop_sequences");
    const auto max_tokens = integer(body, "max_tokens");
    if (!max_tokens && !counting) throw invalid_request("max_tokens: Field required");
    task.params.max_tokens = int(std::clamp<int64_t>(max_tokens.value_or(0), 0, 1 << 30));
    task.stream = boolean(body, "stream").value_or(false);
    return task;
}

App::ChatRun App::run_chat(ChatTask & task, const char * id_prefix, Response & response, std::function<void()> on_start,
                           std::function<void(int, int)> on_prompt,
                           std::function<bool(const ChatDelta &, const std::vector<GeneratedToken> &)> on_delta) {
    ChatRun run;
    ChatParser parser(task.prompt.thinking, &task.options.tools, task.inline_reasoning);
    const auto absorb = [&](ChatDelta delta, const std::vector<GeneratedToken> & tokens) {
        for (auto & call : delta.tool_calls) call.id = id_prefix + random_token(24);
        run.reasoning += delta.reasoning; run.content += delta.content;
        run.calls.insert(run.calls.end(), delta.tool_calls.begin(), delta.tool_calls.end());
        return !on_delta || on_delta(delta, tokens);
    };
    absorb(parser.feed(task.prompt.reply_prefix), {});
    GenerationCallbacks callbacks;
    callbacks.on_start = std::move(on_start);
    callbacks.on_prompt = std::move(on_prompt);
    callbacks.on_text = [&](std::string_view piece, const std::vector<GeneratedToken> & tokens) { return absorb(parser.feed(piece), tokens); };
    callbacks.cancelled = [&] { return !response.alive(); };
    run.result = runtime_.generate(task.tokens, task.params, callbacks);
    absorb(parser.finish(), {});
    return run;
}

void App::chat_completions(const Request & request, Response & response) {
    auto task = openai_task(parse_body(request));
    prepare(task);
    const std::string id = "chatcmpl-" + random_token(24);
    const int64_t created = unix_time();
    const auto finish_reason = [](const ChatRun & run) {
        return !run.calls.empty() && run.result.stop != StopType::Limit ? "tool_calls" : run.result.stop == StopType::Limit ? "length" : "stop";
    };
    const auto logprobs = [&](const std::vector<GeneratedToken> & tokens) {
        Json content = Json::array();
        for (const auto & token : tokens) content.push(token_logprob(token));
        return Json::object({{"content", std::move(content)}});
    };
    if (!task.stream) {
        const auto run = run_chat(task, "call_", response, {}, {}, {});
        Json message = Json::object({{"role", "assistant"}, {"content", run.content.empty() && !run.calls.empty() ? Json() : Json(run.content)}});
        if (!run.reasoning.empty()) message["reasoning_content"] = run.reasoning;
        if (!run.calls.empty()) message["tool_calls"] = tool_calls_json(run.calls, 0, false);
        Json choice = Json::object({{"index", 0}, {"message", std::move(message)},
            {"logprobs", task.params.logprobs ? logprobs(run.result.tokens) : Json()}, {"finish_reason", finish_reason(run)}});
        send_json(response, 200, Json::object({{"id", id}, {"object", "chat.completion"}, {"created", created},
            {"model", task.model}, {"system_fingerprint", version}, {"choices", list({std::move(choice)})},
            {"usage", openai_usage(run.result)}, {"timings", timings(run.result)}}));
        return;
    }
    const auto chunk = [&](Json delta, Json finish) {
        return Json::object({{"id", id}, {"object", "chat.completion.chunk"}, {"created", created}, {"model", task.model},
            {"system_fingerprint", version},
            {"choices", list({Json::object({{"index", 0}, {"delta", std::move(delta)}, {"finish_reason", std::move(finish)}})})}});
    };
    size_t call_index = 0;
    std::vector<GeneratedToken> unreported;
    const auto run = run_chat(task, "call_", response,
        [&] { response.begin(200, "text/event-stream"); },
        [&](int, int) { sse(response, chunk(Json::object({{"role", "assistant"}, {"content", Json()}}), Json())); },
        [&](const ChatDelta & delta, const std::vector<GeneratedToken> & tokens) {
            if (task.params.logprobs) unreported.insert(unreported.end(), tokens.begin(), tokens.end());
            const auto send = [&](Json body) {
                Json event = chunk(std::move(body), Json());
                if (task.params.logprobs && !unreported.empty()) {
                    event["choices"].items()[0]["logprobs"] = logprobs(unreported);
                    unreported.clear();
                }
                return sse(response, event);
            };
            bool ok = true;
            if (!delta.reasoning.empty()) ok = send(Json::object({{"reasoning_content", delta.reasoning}}));
            if (ok && !delta.content.empty()) ok = send(Json::object({{"content", delta.content}}));
            if (ok && !delta.tool_calls.empty()) {
                ok = send(Json::object({{"tool_calls", tool_calls_json(delta.tool_calls, call_index, true)}}));
                call_index += delta.tool_calls.size();
            }
            return ok;
        });
    if (!response.started()) return; // Cancelled while queued.
    Json last = chunk(Json::object(), finish_reason(run));
    if (!unreported.empty()) last["choices"].items()[0]["logprobs"] = logprobs(unreported);
    last["timings"] = timings(run.result);
    sse(response, last);
    if (task.include_usage) {
        Json usage = chunk(Json::object(), Json());
        usage["choices"] = Json::array();
        usage["usage"] = openai_usage(run.result);
        sse(response, usage);
    }
    response.write("data: [DONE]\n\n");
}

void App::completions(const Request & request, Response & response) {
    const Json body = parse_body(request);
    const auto * prompt = present(body, "prompt");
    if (!prompt) throw invalid_request("'prompt' is required");
    const auto tokens = prompt_tokens(*prompt);
    GenerationParams params;
    sampling(body, params);
    params.stop = stop_words(body, "stop");
    params.max_tokens = config_.max_tokens;
    for (const char * key : {"n_predict", "max_tokens"})
        if (const auto value = integer(body, key)) params.max_tokens = int(std::clamp<int64_t>(*value, -1, 1 << 30));
    params.cache_prompt = boolean(body, "cache_prompt").value_or(true);
    if (const auto value = integer(body, "logprobs")) {
        if (*value < 0 || *value > 20) throw invalid_request("'logprobs' must be within 0-20");
        params.logprobs = true; params.top_logprobs = int(*value);
    }
    if (const auto n = integer(body, "n"); n && *n != 1) throw invalid_request("Only n = 1 is supported");
    if (const auto suffix = text(body, "suffix"); suffix && !suffix->empty()) throw invalid_request("'suffix' is not supported; use /infill");
    const bool stream = boolean(body, "stream").value_or(false), echo = boolean(body, "echo").value_or(false);
    const std::string model = text(body, "model").value_or(model_name()), id = "cmpl-" + random_token(24);
    const int64_t created = unix_time();
    const auto & tokenizer = runtime_.tokenizer();
    size_t offset = 0;
    const auto logprobs = [&](const std::vector<GeneratedToken> & generated) {
        Json pieces = Json::array(), values = Json::array(), top = Json::array(), offsets = Json::array();
        for (const auto & token : generated) {
            const auto piece = sanitize_utf8(tokenizer.piece(token.id, true));
            Json alternatives = Json::object();
            for (const auto & candidate : token.top) alternatives[sanitize_utf8(tokenizer.piece(candidate.id, true))] = candidate.logprob;
            offsets.push(offset); offset += piece.size();
            pieces.push(piece); values.push(token.logprob); top.push(std::move(alternatives));
        }
        return Json::object({{"tokens", std::move(pieces)}, {"token_logprobs", std::move(values)},
                             {"top_logprobs", std::move(top)}, {"text_offset", std::move(offsets)}});
    };
    const auto event = [&](std::string content, Json probabilities, Json finish) {
        return Json::object({{"id", id}, {"object", "text_completion"}, {"created", created}, {"model", model},
            {"system_fingerprint", version},
            {"choices", list({Json::object({{"text", std::move(content)}, {"index", 0}, {"logprobs", std::move(probabilities)},
                                            {"finish_reason", std::move(finish)}})})}});
    };
    const auto finish_reason = [](const GenerationResult & r) { return r.stop == StopType::Limit ? "length" : "stop"; };
    const std::string prefix = echo ? tokenizer.decode(tokens, false) : std::string();
    GenerationCallbacks callbacks;
    callbacks.cancelled = [&] { return !response.alive(); };
    if (!stream) {
        const auto result = runtime_.generate(tokens, params, callbacks);
        Json out = event(prefix + result.text, params.logprobs ? logprobs(result.tokens) : Json(), finish_reason(result));
        out["usage"] = openai_usage(result); out["timings"] = timings(result);
        send_json(response, 200, out);
        return;
    }
    callbacks.on_start = [&] { response.begin(200, "text/event-stream"); if (echo) sse(response, event(prefix, Json(), Json())); };
    callbacks.on_text = [&](std::string_view piece, const std::vector<GeneratedToken> & generated) {
        return sse(response, event(std::string(piece), params.logprobs ? logprobs(generated) : Json(), Json()));
    };
    const auto result = runtime_.generate(tokens, params, callbacks);
    if (!response.started()) return;
    Json last = event("", Json(), finish_reason(result));
    last["usage"] = openai_usage(result); last["timings"] = timings(result);
    sse(response, last);
    response.write("data: [DONE]\n\n");
}

void App::messages(const Request & request, Response & response) {
    auto task = anthropic_task(parse_body(request), false);
    prepare(task);
    const std::string id = "msg_" + random_token(24);
    const auto stop_reason = [](const ChatRun & run) {
        return run.result.stop == StopType::Limit ? "max_tokens" : !run.calls.empty() ? "tool_use"
             : run.result.stop == StopType::Word ? "stop_sequence" : "end_turn";
    };
    const auto stop_sequence = [](const ChatRun & run) {
        return run.result.stop == StopType::Word && run.calls.empty() ? Json(run.result.stopping_word) : Json();
    };
    if (!task.stream) {
        const auto run = run_chat(task, "toolu_", response, {}, {}, {});
        Json content = Json::array();
        if (!run.reasoning.empty()) content.push(Json::object({{"type", "thinking"}, {"thinking", run.reasoning}, {"signature", ""}}));
        if (!run.content.empty() || (run.calls.empty() && run.reasoning.empty())) content.push(Json::object({{"type", "text"}, {"text", run.content}}));
        for (const auto & call : run.calls)
            content.push(Json::object({{"type", "tool_use"}, {"id", call.id}, {"name", call.name}, {"input", call.arguments}}));
        send_json(response, 200, Json::object({{"id", id}, {"type", "message"}, {"role", "assistant"}, {"model", task.model},
            {"content", std::move(content)}, {"stop_reason", stop_reason(run)}, {"stop_sequence", stop_sequence(run)},
            {"usage", anthropic_usage(run.result)}}));
        return;
    }
    enum class Block { None, Thinking, Text } open = Block::None;
    int index = -1;
    const auto close = [&] {
        if (open == Block::None) return true;
        if (open == Block::Thinking)
            sse(response, "content_block_delta", Json::object({{"type", "content_block_delta"}, {"index", index},
                {"delta", Json::object({{"type", "signature_delta"}, {"signature", ""}})}}));
        open = Block::None;
        return sse(response, "content_block_stop", Json::object({{"type", "content_block_stop"}, {"index", index}}));
    };
    const auto start = [&](Json block) {
        return sse(response, "content_block_start", Json::object({{"type", "content_block_start"}, {"index", ++index},
            {"content_block", std::move(block)}}));
    };
    const auto delta_event = [&](Json delta) {
        return sse(response, "content_block_delta", Json::object({{"type", "content_block_delta"}, {"index", index}, {"delta", std::move(delta)}}));
    };
    const auto run = run_chat(task, "toolu_", response,
        [&] { response.begin(200, "text/event-stream"); sse(response, "ping", Json::object({{"type", "ping"}})); },
        [&](int prompt, int cached) {
            GenerationResult usage; usage.prompt_tokens = prompt; usage.cached_tokens = cached;
            sse(response, "message_start", Json::object({{"type", "message_start"}, {"message", Json::object({{"id", id},
                {"type", "message"}, {"role", "assistant"}, {"model", task.model}, {"content", Json::array()},
                {"stop_reason", Json()}, {"stop_sequence", Json()}, {"usage", anthropic_usage(usage)}})}}));
        },
        [&](const ChatDelta & delta, const std::vector<GeneratedToken> &) {
            bool ok = true;
            if (!delta.reasoning.empty()) {
                if (open != Block::Thinking) { close(); start(Json::object({{"type", "thinking"}, {"thinking", ""}, {"signature", ""}})); open = Block::Thinking; }
                ok = delta_event(Json::object({{"type", "thinking_delta"}, {"thinking", delta.reasoning}}));
            }
            if (ok && !delta.content.empty()) {
                if (open != Block::Text) { close(); start(Json::object({{"type", "text"}, {"text", ""}})); open = Block::Text; }
                ok = delta_event(Json::object({{"type", "text_delta"}, {"text", delta.content}}));
            }
            for (const auto & call : delta.tool_calls) {
                close();
                start(Json::object({{"type", "tool_use"}, {"id", call.id}, {"name", call.name}, {"input", Json::object()}}));
                delta_event(Json::object({{"type", "input_json_delta"}, {"partial_json", call.arguments.dump()}}));
                ok = sse(response, "content_block_stop", Json::object({{"type", "content_block_stop"}, {"index", index}}));
            }
            return ok;
        });
    if (!response.started()) return;
    close();
    sse(response, "message_delta", Json::object({{"type", "message_delta"},
        {"delta", Json::object({{"stop_reason", stop_reason(run)}, {"stop_sequence", stop_sequence(run)}})},
        {"usage", anthropic_usage(run.result)}}));
    sse(response, "message_stop", Json::object({{"type", "message_stop"}}));
}

void App::count_tokens(const Request & request, Response & response) {
    auto task = anthropic_task(parse_body(request), true);
    prepare(task);
    send_json(response, 200, Json::object({{"input_tokens", task.tokens.size()}}));
}

void App::native_completion(const Request & request, Response & response, bool infill) {
    const Json body = parse_body(request);
    const auto & tokenizer = runtime_.tokenizer();
    std::vector<int32_t> tokens;
    if (infill) {
        const int32_t prefix = tokenizer.added_token("<|fim_prefix|>"), suffix = tokenizer.added_token("<|fim_suffix|>"),
                      middle = tokenizer.added_token("<|fim_middle|>");
        if (prefix < 0 || suffix < 0 || middle < 0) throw ApiError(501, "not_supported_error", "The model has no fill-in-the-middle tokens");
        const auto part = [&](const char * key) { return tokenizer.encode(text(body, key).value_or(""), false); };
        const auto before = part("input_prefix"), after = part("input_suffix");
        tokens.push_back(prefix); tokens.insert(tokens.end(), before.begin(), before.end());
        tokens.push_back(suffix); tokens.insert(tokens.end(), after.begin(), after.end());
        tokens.push_back(middle);
    } else {
        const auto * prompt = present(body, "prompt");
        if (!prompt) throw invalid_request("'prompt' is required");
        tokens = prompt_tokens(*prompt);
    }
    GenerationParams params;
    sampling(body, params);
    params.stop = stop_words(body, "stop");
    params.max_tokens = config_.max_tokens;
    for (const char * key : {"max_tokens", "n_predict"})
        if (const auto value = integer(body, key)) params.max_tokens = int(std::clamp<int64_t>(*value, -1, 1 << 30));
    params.cache_prompt = boolean(body, "cache_prompt").value_or(true);
    params.render_special = boolean(body, "special").value_or(false);
    if (const auto value = integer(body, "n_probs"); value && *value > 0) {
        params.logprobs = true; params.top_logprobs = int(std::min<int64_t>(*value, 100));
    }
    for (const char * key : {"grammar", "json_schema"})
        if (present(body, key)) throw invalid_request(std::string("'") + key + "' is not supported: constrained output is not implemented");
    const bool stream = boolean(body, "stream").value_or(false);
    const bool return_tokens = boolean(body, "return_tokens").value_or(false) || stream;
    const std::string model = model_name();
    const auto ids = [](const std::vector<GeneratedToken> & generated) {
        Json out = Json::array();
        for (const auto & token : generated) out.push(token.id);
        return out;
    };
    const auto probabilities = [&](const std::vector<GeneratedToken> & generated) {
        Json out = Json::array();
        for (const auto & token : generated) out.push(token_logprob(token));
        return out;
    };
    GenerationCallbacks callbacks;
    callbacks.cancelled = [&] { return !response.alive(); };
    int evaluated = int(tokens.size());
    size_t predicted = 0;
    if (stream) {
        callbacks.on_start = [&] { response.begin(200, "text/event-stream"); };
        callbacks.on_text = [&](std::string_view piece, const std::vector<GeneratedToken> & generated) {
            predicted += generated.size();
            Json event = Json::object({{"index", 0}, {"content", piece}, {"tokens", ids(generated)}, {"stop", false},
                {"id_slot", 0}, {"tokens_predicted", predicted}, {"tokens_evaluated", evaluated}});
            if (params.logprobs) event["completion_probabilities"] = probabilities(generated);
            return sse(response, event);
        };
    }
    const auto result = runtime_.generate(tokens, params, callbacks);
    if (stream && !response.started()) return;
    const char * stop_type = result.stop == StopType::Eos ? "eos" : result.stop == StopType::Limit ? "limit"
                           : result.stop == StopType::Word ? "word" : "none";
    Json out = Json::object({{"index", 0}, {"content", stream ? std::string() : result.text},
        {"tokens", return_tokens && !stream ? ids(result.tokens) : Json::array()}, {"id_slot", 0}, {"stop", true}, {"model", model},
        {"tokens_predicted", result.tokens.size()}, {"tokens_evaluated", result.prompt_tokens},
        {"generation_settings", sampling_json(params)}, {"has_new_line", result.text.find('\n') != std::string::npos},
        {"truncated", false}, {"stop_type", stop_type}, {"stopping_word", result.stopping_word},
        {"stopped_eos", result.stop == StopType::Eos}, {"stopped_word", result.stop == StopType::Word},
        {"stopped_limit", result.stop == StopType::Limit}, {"tokens_cached", result.cached_tokens},
        {"timings", timings(result)}});
    if (params.logprobs && !stream) out["completion_probabilities"] = probabilities(result.tokens);
    if (stream) sse(response, out); else send_json(response, 200, out);
}

void App::tokenize(const Request & request, Response & response) {
    const Json body = parse_body(request);
    const auto & tokenizer = runtime_.tokenizer();
    const bool parse_special = boolean(body, "parse_special").value_or(true);
    std::vector<int32_t> tokens;
    if (boolean(body, "add_special").value_or(false) && tokenizer.bos() >= 0) tokens.push_back(tokenizer.bos());
    const auto & content = body.at("content");
    const auto add = [&](const Json & item) {
        if (item.is_string()) {
            const auto part = tokenizer.encode(item.string(), parse_special);
            tokens.insert(tokens.end(), part.begin(), part.end());
        } else if (item.is_int() && item.integer() >= 0 && item.integer() < tokenizer.size()) tokens.push_back(int32_t(item.integer()));
        else throw invalid_request("'content' must be a string or an array of strings and token ids");
    };
    if (content.is_array()) for (const auto & item : content.items()) add(item);
    else add(content);
    Json out = Json::array();
    const bool pieces = boolean(body, "with_pieces").value_or(false);
    for (const auto token : tokens) {
        if (!pieces) { out.push(token); continue; }
        // A piece that is not valid UTF-8 on its own is given as bytes.
        const auto piece = tokenizer.piece(token, true);
        out.push(Json::object({{"id", token}, {"piece", sanitize_utf8(piece) == piece ? Json(piece) : bytes_json(piece)}}));
    }
    send_json(response, 200, Json::object({{"tokens", std::move(out)}}));
}

void App::detokenize(const Request & request, Response & response) {
    const Json body = parse_body(request);
    const auto & items = body.at("tokens");
    if (!items.is_array()) throw invalid_request("'tokens' must be an array of token ids");
    std::vector<int32_t> tokens;
    for (const auto & item : items.items()) {
        if (!item.is_int() || item.integer() < 0 || item.integer() >= runtime_.tokenizer().size())
            throw invalid_request("'tokens' must contain token ids inside the vocabulary");
        tokens.push_back(int32_t(item.integer()));
    }
    send_json(response, 200, Json::object({{"content", runtime_.tokenizer().decode(tokens, true)}}));
}

void App::apply_template(const Request & request, Response & response) {
    auto task = openai_task(parse_body(request));
    prepare(task);
    send_json(response, 200, Json::object({{"prompt", task.prompt.text}}));
}

void App::health(const Request &, Response & response) {
    if (runtime_.state() == Runtime::State::Ready) send_json(response, 200, Json::object({{"status", "ok"}}));
    else if (runtime_.state() == Runtime::State::Loading) send_json(response, 503, openai_error(503, "unavailable_error", "Loading model"));
    else send_json(response, 500, openai_error(500, "server_error", "Model failed to load: " + runtime_.error()));
}

void App::models(const Request &, Response & response) {
    const auto & info = runtime_.model_info();
    const std::string name = model_name();
    Json model = Json::object({{"id", name}, {"object", "model"}, {"type", "model"}, {"display_name", name},
        {"created", started_}, {"owned_by", "tidal"},
        {"meta", Json::object({{"n_vocab", runtime_.model().vocab}, {"n_ctx_train", runtime_.context()},
                               {"n_params", info.parameters}, {"size", info.stored_weight_bytes}})}});
    send_json(response, 200, Json::object({{"object", "list"}, {"data", list({std::move(model)})},
        {"has_more", false}, {"first_id", name}, {"last_id", name}}));
}

void App::props(const Request &, Response & response) {
    const auto & tokenizer = runtime_.tokenizer();
    GenerationParams params;
    params.sampling = default_sampling();
    params.max_tokens = config_.max_tokens;
    const auto token_text = [&](int32_t token) { return token < 0 ? std::string() : sanitize_utf8(tokenizer.piece(token, true)); };
    send_json(response, 200, Json::object({
        {"default_generation_settings", Json::object({{"id", 0}, {"n_ctx", runtime_.context()}, {"params", sampling_json(params)}})},
        {"total_slots", 1}, {"model_alias", model_name()}, {"model_path", runtime_.config().model},
        {"modalities", Json::object({{"vision", false}, {"audio", false}})},
        {"chat_template", tokenizer.metadata().chat_template}, {"chat_format", runtime_.chat().name()},
        {"bos_token", token_text(tokenizer.bos())}, {"eos_token", token_text(tokenizer.eos())},
        {"build_info", version}, {"backend", runtime_.backend()}, {"n_vocab", runtime_.model().vocab},
        {"tokenizer", tokenizer.byte_fallback() ? "bytes" : "bpe"},
        {"speculative", Json::object({{"draft_tokens", runtime_.config().draft_tokens}})},
        {"thinking_default", config_.thinking.value_or(runtime_.chat().thinking_default())}}));
}

void App::slots(const Request &, Response & response) {
    GenerationParams params;
    params.sampling = default_sampling();
    params.max_tokens = config_.max_tokens;
    send_json(response, 200, list({Json::object({{"id", 0}, {"n_ctx", runtime_.context()},
        {"speculative", runtime_.config().draft_tokens > 0}, {"is_processing", runtime_.busy()},
        {"queued", runtime_.queued()}, {"params", sampling_json(params)},
        {"next_token", Json::object({{"n_decoded", runtime_.decoded()}, {"n_remain", runtime_.remaining()}})}})}));
}

void App::slot_action(const Request & request, Response & response) {
    const auto action = request.parameter("action");
    if (request.path != "/slots/0") throw ApiError(404, "not_found_error", "Unknown slot: there is a single slot with id 0");
    if (action != "erase") throw ApiError(501, "not_supported_error", "Only action=erase is supported");
    runtime_.erase();
    send_json(response, 200, Json::object({{"id_slot", 0}, {"n_erased", 0}}));
}

void App::metrics(const Request &, Response & response) {
    const auto m = runtime_.metrics();
    std::string out;
    const auto metric = [&](const char * name, const char * kind, const char * help, double value) {
        char number[64];
        std::snprintf(number, sizeof number, "%.10g", value);
        out += std::string("# HELP ") + name + " " + help + "\n# TYPE " + name + " " + kind + "\n" + name + " " + number + "\n";
    };
    metric("llamacpp:prompt_tokens_total", "counter", "Number of prompt tokens processed.", double(m.prompt_tokens));
    metric("llamacpp:prompt_seconds_total", "counter", "Prompt process time", m.prompt_seconds);
    metric("llamacpp:tokens_predicted_total", "counter", "Number of generation tokens processed.", double(m.generated_tokens));
    metric("llamacpp:tokens_predicted_seconds_total", "counter", "Predict process time", m.generation_seconds);
    metric("llamacpp:prompt_tokens_seconds", "gauge", "Average prompt throughput in tokens/s.",
           m.prompt_seconds > 0 ? double(m.prompt_tokens) / m.prompt_seconds : 0);
    metric("llamacpp:predicted_tokens_seconds", "gauge", "Average generation throughput in tokens/s.",
           m.generation_seconds > 0 ? double(m.generated_tokens) / m.generation_seconds : 0);
    metric("llamacpp:requests_processing", "gauge", "Number of requests processing.", runtime_.busy() ? 1 : 0);
    metric("llamacpp:requests_deferred", "gauge", "Number of requests deferred.", runtime_.queued());
    metric("tidal:requests_total", "counter", "Number of completed generation requests.", double(m.requests));
    metric("tidal:prompt_tokens_cached_total", "counter", "Number of prompt tokens reused from the prompt cache.", double(m.cached_tokens));
    response.send(200, "text/plain; version=0.0.4; charset=utf-8", out);
}

} // namespace tidal::server
