#include "chat.hpp"

#include <algorithm>

namespace tidal::server {
namespace {

const char * const whitespace = " \t\n\r\f\v";
std::string_view trim(std::string_view text) {
    const auto first = text.find_first_not_of(whitespace);
    if (first == std::string_view::npos) return {};
    return text.substr(first, text.find_last_not_of(whitespace) - first + 1);
}
bool starts_with(std::string_view text, std::string_view prefix) { return text.substr(0, prefix.size()) == prefix; }
bool ends_with(std::string_view text, std::string_view suffix) {
    return text.size() >= suffix.size() && text.substr(text.size() - suffix.size()) == suffix;
}

const char * const tool_instructions =
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested "
    "within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
    "- If there is no function call available, answer the question like normal with your current knowledge and do "
    "not tell the user about function calls\n</IMPORTANT>";

const char * const effort_xhigh =
    "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider "
    "plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
const char * const effort_low =
    "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion "
    "without unnecessary elaboration.";

// Python's str() of a JSON scalar, as the Qwen3.5 template's `| string` renders it.
std::string python_string(const Json & value) {
    if (value.is_null()) return "None";
    if (value.is_bool()) return value.boolean() ? "True" : "False";
    return value.dump(true);
}

} // namespace

ChatFormat::ChatFormat(const std::string & chat_template, const std::string & name) {
    if (name == "chatml") kind_ = Kind::ChatML;
    else if (name == "qwen35") kind_ = Kind::Qwen35;
    else if (name == "qwen38") kind_ = Kind::Qwen38;
    else if (name != "auto") throw std::invalid_argument("--chat-template must be auto, chatml, qwen35 or qwen38");
    else if (chat_template.find("<function=") == std::string::npos || chat_template.find("<|im_start|>") == std::string::npos)
        kind_ = Kind::ChatML;
    else kind_ = chat_template.find("reasoning_effort") != std::string::npos ? Kind::Qwen38 : Kind::Qwen35;
}
const char * ChatFormat::name() const {
    return kind_ == Kind::Qwen38 ? "qwen38" : kind_ == Kind::Qwen35 ? "qwen35" : "chatml";
}

ChatPrompt ChatFormat::render(const std::vector<Message> & all, const ChatOptions & options) const {
    if (all.empty()) throw ChatError("No messages provided.");
    ChatPrompt prompt;
    std::string & out = prompt.text;
    // A trailing assistant message is continued instead of closed.
    const bool resume = options.add_generation_prompt && all.back().role == "assistant" && all.back().tool_calls.empty();
    const size_t count = all.size() - (resume ? 1 : 0);
    if (!count) throw ChatError("No user query found in messages.");
    const bool forced = !options.forced_tool.empty() && supports_tools() && options.add_generation_prompt && !resume;
    auto thinking = options.enable_thinking;
    if (resume || forced) thinking = false;

    if (kind_ == Kind::ChatML) {
        for (size_t i = 0; i < count; ++i) {
            const auto & message = all[i];
            out += "<|im_start|>" + message.role + "\n" + std::string(trim(message.content)) + "<|im_end|>\n";
        }
        if (options.add_generation_prompt) out += "<|im_start|>assistant\n";
        if (resume) out += all.back().content;
        return prompt;
    }

    const bool v38 = kind_ == Kind::Qwen38;
    std::string instructions;
    if (v38 && thinking.value_or(true)) {
        const std::string effort = options.reasoning_effort.value_or("xhigh");
        if (effort == "xhigh") instructions = effort_xhigh;
        else if (effort == "low") instructions = effort_low;
        else if (effort != "medium")
            throw ChatError("Unexpected reasoning effort " + effort + ". Supported types are xhigh (default), medium, and low.");
    }
    const bool system = all[0].role == "system";
    const std::string system_text(system ? trim(all[0].content) : std::string_view());
    if (options.tools.is_array() && options.tools.size()) {
        out += "<|im_start|>system\n";
        if (!instructions.empty()) out += instructions + "\n\n";
        out += "# Tools\n\nYou have access to the following functions:\n\n<tools>";
        for (const auto & tool : options.tools.items()) { out += "\n"; out += tool.dump(true); }
        out += "\n</tools>";
        out += tool_instructions;
        if (!system_text.empty()) out += "\n\n" + system_text;
        out += "<|im_end|>\n";
    } else if (v38) {
        if (!system_text.empty()) out += "<|im_start|>system\n" + (instructions.empty() ? "" : instructions + "\n\n") + system_text + "<|im_end|>\n";
        else if (!instructions.empty()) out += "<|im_start|>system\n" + instructions + "<|im_end|>\n";
    } else if (system) out += "<|im_start|>system\n" + system_text + "<|im_end|>\n";

    // The last user message that is a query, not a tool response.
    size_t last_query = count - 1;
    bool found = false;
    for (size_t i = count; i-- > 0 && !found;) {
        if (all[i].role != "user") continue;
        const auto content = trim(all[i].content);
        if (!(starts_with(content, "<tool_response>") && ends_with(content, "</tool_response>"))) { found = true; last_query = i; }
    }
    if (!found) throw ChatError("No user query found in messages.");

    for (size_t i = 0; i < count; ++i) {
        const auto & message = all[i];
        std::string content(trim(message.content));
        if (message.role == "system") {
            if (i) throw ChatError("System message must be at the beginning.");
        } else if (message.role == "user") {
            out += "<|im_start|>user\n" + content + "<|im_end|>\n";
        } else if (message.role == "assistant") {
            std::string reasoning;
            if (message.reasoning) reasoning = *message.reasoning;
            else if (!v38 && content.find("</think>") != std::string::npos) {
                // content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n')
                std::string_view head = std::string_view(content).substr(0, content.find("</think>"));
                while (!head.empty() && head.back() == '\n') head.remove_suffix(1);
                if (const auto open = head.rfind("<think>"); open != std::string_view::npos) head.remove_prefix(open + 7);
                while (!head.empty() && head.front() == '\n') head.remove_prefix(1);
                reasoning = std::string(head);
                std::string_view tail = std::string_view(content).substr(content.rfind("</think>") + 8);
                while (!tail.empty() && tail.front() == '\n') tail.remove_prefix(1);
                content = std::string(tail);
            }
            reasoning = std::string(trim(reasoning));
            const bool keep = i > last_query || (v38 && options.preserve_thinking.value_or(true));
            out += "<|im_start|>assistant\n";
            if (keep) out += "<think>\n" + reasoning + "\n</think>\n\n";
            out += content;
            for (size_t k = 0; k < message.tool_calls.size(); ++k) {
                const auto & call = message.tool_calls[k];
                out += k ? "\n" : trim(content).empty() ? "" : "\n\n";
                out += "<tool_call>\n<function=" + call.name + ">\n";
                if (call.arguments.is_object()) for (const auto & [name, value] : call.arguments.members()) {
                    out += "<parameter=" + name + ">\n";
                    if (value.is_string()) out += value.string();
                    else if (v38 || value.is_object() || value.is_array()) out += value.dump(true);
                    else out += python_string(value);
                    out += "\n</parameter>\n";
                }
                out += "</function>\n</tool_call>";
            }
            out += "<|im_end|>\n";
        } else if (message.role == "tool") {
            if (i && all[i - 1].role != "tool") out += "<|im_start|>user";
            out += "\n<tool_response>\n" + content + "\n</tool_response>";
            if (i + 1 == count || all[i + 1].role != "tool") out += "<|im_end|>\n";
        } else throw ChatError("Unexpected message role.");
    }
    if (options.add_generation_prompt) {
        out += "<|im_start|>assistant\n";
        prompt.thinking = v38 ? thinking.value_or(true) : thinking.value_or(false);
        out += prompt.thinking ? "<think>\n" : "<think>\n\n</think>\n\n";
        if (resume) out += all.back().content;
        if (forced) {
            prompt.reply_prefix = "<tool_call>\n<function=";
            if (options.forced_tool != "*") prompt.reply_prefix += options.forced_tool + ">\n";
            out += prompt.reply_prefix;
        }
    }
    return prompt;
}

ChatParser::ChatParser(bool thinking, const Json * tools, bool inline_reasoning)
    : state_(thinking ? State::Reasoning : State::Content),
      tools_(tools && tools->is_array() && tools->size() ? tools : nullptr), inline_reasoning_(inline_reasoning) {}

// Appends to the current section. Trailing whitespace is held back until more
// text follows, so that the whitespace before a closing tag or a tool call is
// dropped; `final` releases it for ordinary content.
void ChatParser::emit(ChatDelta & delta, std::string_view text, bool final) {
    if (text.empty() && !final) return;
    std::string & target = state_ == State::Reasoning && !inline_reasoning_ ? delta.reasoning : delta.content;
    if (skip_space_) {
        const auto first = text.find_first_not_of(whitespace);
        if (first == std::string_view::npos) return;
        text.remove_prefix(first); skip_space_ = false;
    }
    const bool hold = state_ == State::Reasoning || tools_;
    if (!hold) { target += text; return; }
    const auto last = text.find_last_not_of(whitespace);
    if (last == std::string_view::npos) space_ += text;
    else {
        target += space_; target += text.substr(0, last + 1);
        space_ = std::string(text.substr(last + 1));
    }
    if (final && state_ == State::Content) { target += space_; space_.clear(); }
}

// Length of the longest suffix of `text` that is a proper prefix of `tag`.
static size_t partial_tag(std::string_view text, std::string_view tag) {
    for (size_t n = std::min(text.size(), tag.size() - 1); n > 0; --n)
        if (text.substr(text.size() - n) == tag.substr(0, n)) return n;
    return 0;
}

void ChatParser::step(ChatDelta & delta, bool final) {
    for (;;) {
        if (state_ == State::Reasoning) {
            if (at_start_ && inline_reasoning_) delta.content += "<think>\n";
            at_start_ = false;
            const auto close = buffer_.find("</think>");
            if (close == std::string::npos) {
                const size_t keep = final ? 0 : partial_tag(buffer_, "</think>");
                emit(delta, std::string_view(buffer_).substr(0, buffer_.size() - keep), false);
                buffer_.erase(0, buffer_.size() - keep);
                if (final) space_.clear();
                return;
            }
            emit(delta, std::string_view(buffer_).substr(0, close), false);
            space_.clear();
            if (inline_reasoning_) delta.content += "\n</think>\n\n";
            buffer_.erase(0, close + 8);
            state_ = State::Content; skip_space_ = true;
            continue;
        }
        if (state_ == State::Content) {
            if (at_start_) {
                // A reply may open a thinking section even when none was prompted.
                const auto first = buffer_.find_first_not_of(whitespace);
                const std::string_view head = first == std::string::npos ? std::string_view() : std::string_view(buffer_).substr(first);
                if (!final && (head.empty() || (head.size() < 7 && starts_with("<think>", head)))) return;
                at_start_ = false;
                if (starts_with(head, "<think>")) {
                    buffer_.erase(0, first + 7);
                    state_ = State::Reasoning; skip_space_ = true;
                    if (inline_reasoning_) delta.content += "<think>\n";
                    continue;
                }
            }
            const auto open = tools_ ? buffer_.find("<tool_call>") : std::string::npos;
            if (open == std::string::npos) {
                const size_t keep = final || !tools_ ? 0 : partial_tag(buffer_, "<tool_call>");
                emit(delta, std::string_view(buffer_).substr(0, buffer_.size() - keep), final);
                buffer_.erase(0, buffer_.size() - keep);
                return;
            }
            emit(delta, std::string_view(buffer_).substr(0, open), false);
            buffer_.erase(0, open + 11);
            state_ = State::Tool;
            continue;
        }
        const auto close = buffer_.find("</tool_call>");
        if (close == std::string::npos && !final) return;
        const std::string body = buffer_.substr(0, close);
        buffer_.erase(0, close == std::string::npos ? buffer_.size() : close + 12);
        state_ = State::Content;
        // Whitespace around a parsed call is dropped; a failed one stays text.
        if (tool_call(body, delta)) { space_.clear(); skip_space_ = true; }
        else { skip_space_ = false; emit(delta, "<tool_call>" + body + (close == std::string::npos ? "" : "</tool_call>"), final); }
        if (close == std::string::npos) return;
    }
}

bool ChatParser::tool_call(std::string_view raw, ChatDelta & delta) const {
    const auto body = trim(raw);
    ToolCall call;
    if (starts_with(body, "{")) {
        // Hermes-style JSON: {"name": ..., "arguments": {...}}
        try {
            const Json value = Json::parse(body);
            if (!value.at("name").is_string()) return false;
            call.name = value.at("name").string();
            const auto & arguments = value.at("arguments");
            if (arguments.is_object()) call.arguments = arguments;
            else if (arguments.is_string()) call.arguments = Json::parse(arguments.string());
            else if (!arguments.is_null()) return false;
        } catch (const JsonError &) { return false; }
        if (!call.arguments.is_object()) return false;
        delta.tool_calls.push_back(std::move(call));
        return true;
    }
    if (!starts_with(body, "<function=")) return false;
    const auto name_end = body.find('>');
    if (name_end == std::string_view::npos) return false;
    call.name = std::string(trim(body.substr(10, name_end - 10)));
    if (call.name.empty()) return false;
    const Json * properties = nullptr;
    if (tools_) for (const auto & tool : tools_->items()) {
        const auto & function = tool.at("function");
        if (function.at("name").is_string() && function.at("name").string() == call.name)
            properties = function.at("parameters").find("properties");
    }
    std::string_view rest = body.substr(name_end + 1);
    if (const auto end = rest.rfind("</function>"); end != std::string_view::npos) rest = rest.substr(0, end);
    for (;;) {
        const auto open = rest.find("<parameter=");
        if (open == std::string_view::npos) break;
        const auto key_end = rest.find('>', open);
        if (key_end == std::string_view::npos) break;
        const std::string key(trim(rest.substr(open + 11, key_end - open - 11)));
        auto close = rest.find("</parameter>", key_end);
        std::string_view value = rest.substr(key_end + 1, close == std::string_view::npos ? close : close - key_end - 1);
        rest = close == std::string_view::npos ? std::string_view() : rest.substr(close + 12);
        if (starts_with(value, "\n")) value.remove_prefix(1);
        if (ends_with(value, "\n")) value.remove_suffix(1);
        if (key.empty()) continue;
        // String parameters are literal; anything else is JSON when it parses.
        const Json * type = properties ? properties->at(key).find("type") : nullptr;
        const bool literal = type && type->is_string() && type->string() == "string";
        Json parsed(value);
        if (!literal) {
            try {
                Json candidate = Json::parse(trim(value));
                if (type || !candidate.is_string()) parsed = std::move(candidate);
            } catch (const JsonError &) {}
        }
        call.arguments[key] = std::move(parsed);
    }
    delta.tool_calls.push_back(std::move(call));
    return true;
}

ChatDelta ChatParser::feed(std::string_view text) {
    ChatDelta delta;
    buffer_ += text;
    step(delta, false);
    return delta;
}
ChatDelta ChatParser::finish() {
    ChatDelta delta;
    step(delta, true);
    return delta;
}

} // namespace tidal::server
