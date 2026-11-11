#pragma once
// Chat prompt construction and model output parsing.
//
// The model's Jinja chat template is not interpreted. The Qwen3.5 and Qwen3.8
// templates are implemented natively and selected by recognizing the template
// stored in the GGUF; anything else falls back to plain ChatML.
#include "json.hpp"

#include <optional>
#include <string>
#include <vector>

namespace tidal::server {

struct ToolCall {
    std::string id, name;
    Json arguments = Json::object();
};

struct Message {
    std::string role, content;
    std::optional<std::string> reasoning;
    std::vector<ToolCall> tool_calls;
};

struct ChatOptions {
    Json tools;                                 // OpenAI-style tool objects; null or empty for none.
    std::optional<bool> enable_thinking;        // Unset: the template's default.
    std::optional<std::string> reasoning_effort; // xhigh | medium | low (Qwen3.8).
    std::optional<bool> preserve_thinking;      // Qwen3.8: keep reasoning of earlier turns.
    bool add_generation_prompt = true;
    // Start the reply inside a tool call: "*" for any tool, otherwise its name.
    std::string forced_tool;
};

struct ChatPrompt {
    std::string text;
    bool thinking = false; // The prompt ends inside an open <think> block.
    // Text the prompt already contains that belongs to the reply's structure
    // (a forced tool call opening); the output parser consumes it first.
    std::string reply_prefix;
};

struct ChatError : std::runtime_error { using std::runtime_error::runtime_error; };

class ChatFormat {
public:
    enum class Kind { ChatML, Qwen35, Qwen38 };
    // `name` is auto (recognize `chat_template`), chatml, qwen35 or qwen38.
    ChatFormat(const std::string & chat_template, const std::string & name);
    Kind kind() const { return kind_; }
    const char * name() const;
    bool supports_tools() const { return kind_ != Kind::ChatML; }
    bool supports_thinking() const { return kind_ != Kind::ChatML; }
    // Whether a reply thinks when the request does not say.
    bool thinking_default() const { return kind_ == Kind::Qwen38; }
    // A trailing assistant message without tool calls is left open so that the
    // reply continues it. Throws ChatError for conversations the template rejects.
    ChatPrompt render(const std::vector<Message> & messages, const ChatOptions & options) const;
private:
    Kind kind_;
};

struct ChatDelta {
    std::string reasoning, content;
    std::vector<ToolCall> tool_calls; // Complete calls; ids are left empty.
    bool empty() const { return reasoning.empty() && content.empty() && tool_calls.empty(); }
};

// Incremental split of generated text into reasoning, content and tool calls.
class ChatParser {
public:
    // `tools` (may be null) enables <tool_call> parsing and types the arguments.
    // `inline_reasoning` leaves the thinking section in the content, tags included.
    ChatParser(bool thinking, const Json * tools, bool inline_reasoning);
    ChatDelta feed(std::string_view text);
    // Flushes held-back text; an unterminated tool call is parsed if possible.
    ChatDelta finish();
private:
    enum class State { Reasoning, Content, Tool };
    State state_;
    const Json * tools_;
    bool inline_reasoning_, at_start_ = true, skip_space_ = false;
    std::string buffer_, space_;
    void step(ChatDelta & delta, bool final);
    void emit(ChatDelta & delta, std::string_view text, bool final);
    bool tool_call(std::string_view body, ChatDelta & delta) const;
};

} // namespace tidal::server
