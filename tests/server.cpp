// Unit tests of the server's text layer: JSON, tokenizer, chat prompt and
// output parsing. The HTTP API is covered by tests/test_server.py.
#include "chat.hpp"
#include "json.hpp"
#include "tokenizer.hpp"

#include "gguf.h"

#include <cstdio>
#include <filesystem>
#include <iostream>

#include <unistd.h>

using namespace tidal::server;

static void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}
template<class E, class F> static void rejects(F && f, const char * what) {
    try { f(); } catch (const E &) { return; }
    throw std::runtime_error(std::string("Expected rejection: ") + what);
}

static void test_json() {
    const std::string source = R"({"a":1,"b":[true,false,null,-2.5,1e3],"c":{"k":"v\n\"\u00e9\ud83d\ude00"},"big":9007199254740993})";
    const Json value = Json::parse(source);
    require(value.at("a").is_int() && value.at("a").integer() == 1, "integer");
    require(value.at("big").integer() == 9007199254740993, "64-bit integers must stay exact");
    require(!value.at("b").items()[3].is_int() && value.at("b").items()[3].number() == -2.5, "double");
    require(value.at("c").at("k").string() == "v\n\"\xC3\xA9\xF0\x9F\x98\x80", "escapes and surrogate pairs");
    require(value.at("missing").is_null() && !value.contains("missing"), "absent key");
    require(Json::parse(value.dump()).dump() == value.dump(), "round trip");
    require(value.members()[0].first == "a" && value.members()[3].first == "big", "key order is preserved");
    require(Json::parse("{\"a\":1,\"a\":2}").at("a").integer() == 2, "last duplicate key wins");
    require(Json::object({{"x", Json()}}).dump() == "{\"x\":null}", "null member");
    const Json nested = Json::object({{"name", "f"}, {"p", Json::object({{"n", 1.5}, {"s", "한"}})}, {"l", Json(Json::Array{1, "a"})}});
    require(nested.dump() == "{\"name\":\"f\",\"p\":{\"n\":1.5,\"s\":\"한\"},\"l\":[1,\"a\"]}", "compact output");
    require(nested.dump(true) == "{\"name\": \"f\", \"p\": {\"n\": 1.5, \"s\": \"한\"}, \"l\": [1, \"a\"]}", "Python separators");
    require(Json(0.8f).dump() == "0.8" && Json(2.0).dump() == "2.0" && Json(1e300 * 1e300).dump() == "null", "numbers");
    require(Json(std::string("\x01\xFF")).dump() == "\"\\u0001\xEF\xBF\xBD\"", "control characters and invalid UTF-8");
    require(Json::parse("\"\\ud800\"").string() == "\xEF\xBF\xBD", "lone surrogate");
    for (const char * bad : {"", "{", "[1,]", "{\"a\":}", "01", "1.", "\"\\x\"", "nul", "[1] 2", "\"\n\"", "{a:1}", "-"})
        rejects<JsonError>([&] { Json::parse(bad); }, bad);
    rejects<JsonError>([] { Json::parse(std::string(300, '[') + std::string(300, ']')); }, "deep nesting");

    require(sanitize_utf8("a\xC3\xA9\xE2\x82") == "a\xC3\xA9\xEF\xBF\xBD" && sanitize_utf8("\xE2\x82z\xF0\x9F\x98") == "\xEF\xBF\xBDz\xEF\xBF\xBD", "a truncated sequence is replaced once");
    require(sanitize_utf8("\xC0\x80\xED\xA0\x80") == std::string("\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD"), "overlong and surrogate");
    require(utf8_complete_prefix("ab\xED\x95") == 2 && utf8_complete_prefix("ab\xED\x95\x9C") == 5, "partial character");
    require(utf8_complete_prefix("ab\xFF") == 3 && utf8_complete_prefix("\xF0\x9F") == 0 && utf8_complete_prefix("") == 0, "prefix edge cases");
}

static std::vector<std::string> pieces(const std::string & text, bool marks) {
    std::vector<std::string> out;
    size_t at = 0;
    for (const size_t length : Tokenizer::split(text, marks)) { out.push_back(text.substr(at, length)); at += length; }
    require(at == text.size(), "split must cover the text");
    return out;
}
static void test_split() {
    using V = std::vector<std::string>;
    require(pieces("Hello world", true) == V({"Hello", " world"}), "words");
    require(pieces("it's I'LL they've", true) == V({"it", "'s", " I", "'LL", " they", "'ve"}), "contractions");
    require(pieces("a  b", true) == V({"a", " ", " b"}), "double space");
    require(pieces("x\n\n  y", true) == V({"x", "\n\n", " ", " y"}), "newlines then indentation");
    require(pieces("a \n b", true) == V({"a", " \n", " b"}), "space before newline");
    require(pieces("a  ", true) == V({"a", "  "}), "trailing spaces");
    require(pieces("12a", true) == V({"1", "2", "a"}), "digits are single");
    require(pieces("a...\n\nb", true) == V({"a", "...\n\n", "b"}), "punctuation takes newlines");
    require(pieces(" !! x", true) == V({" !!", " x"}), "space before punctuation");
    require(pieces("\tword", true) == V({"\tword"}), "one non-letter prefix");
    require(pieces("한글 테스트", true) == V({"한글", " 테스트"}), "Hangul");
    require(pieces("e\xCC\x81x", true) == V({"e\xCC\x81x"}), "qwen35 keeps marks in words");
    require(pieces("e\xCC\x81x", false) == V({"e", "\xCC\x81x"}), "qwen2 treats marks as a prefix");
    require(pieces("a\xE3\x80\x80" "b", true) == V({"a", "\xE3\x80\x80" "b"}), "ideographic space is a prefix");
    require(pieces("", true).empty(), "empty text");
}

// Byte-level alphabet character of `byte`, as in GPT-2 vocabularies.
static std::string alphabet(int byte) {
    static std::vector<std::string> table = [] {
        std::vector<std::string> t(256);
        uint32_t next = 256;
        for (int b = 0; b < 256; ++b) {
            const bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174;
            append_utf8(t[size_t(b)], printable ? uint32_t(b) : next++);
        }
        return t;
    }();
    return table[size_t(byte)];
}

static void test_tokenizer(const std::filesystem::path & directory) {
    std::vector<std::string> tokens;
    std::vector<int32_t> types;
    for (int b = 0; b < 256; ++b) { tokens.push_back(alphabet(b)); types.push_back(1); }
    const std::string space = alphabet(' ');
    const std::vector<std::pair<std::string, std::string>> merges = {
        {"h", "e"}, {"l", "l"}, {"he", "ll"}, {"hell", "o"}, {space, "w"}};
    std::vector<std::string> merge_text;
    for (const auto & [left, right] : merges) { tokens.push_back(left + right); types.push_back(1); merge_text.push_back(left + " " + right); }
    const int hello = 256 + 3, space_w = 256 + 4, ll = 256 + 1;
    const int end = int(tokens.size()); tokens.push_back("<|im_end|>"); types.push_back(3);
    const int think = int(tokens.size()); tokens.push_back("<think>"); types.push_back(4);
    const int unused = int(tokens.size()); tokens.push_back("[PAD]"); types.push_back(5);
    const auto strings = [](const std::vector<std::string> & values) {
        std::vector<const char *> out;
        for (const auto & value : values) out.push_back(value.c_str());
        return out;
    };
    const auto path = (directory / "tokenizer.gguf").string();
    {
        gguf_context * g = gguf_init_empty();
        gguf_set_val_str(g, "general.name", "unit");
        gguf_set_val_str(g, "tokenizer.ggml.model", "gpt2");
        gguf_set_val_str(g, "tokenizer.ggml.pre", "qwen35");
        gguf_set_val_str(g, "tokenizer.chat_template", "x");
        gguf_set_val_f32(g, "general.sampling.temp", 0.5f);
        gguf_set_val_u32(g, "tokenizer.ggml.eos_token_id", uint32_t(end));
        auto token_pointers = strings(tokens), merge_pointers = strings(merge_text);
        gguf_set_arr_str(g, "tokenizer.ggml.tokens", token_pointers.data(), token_pointers.size());
        gguf_set_arr_str(g, "tokenizer.ggml.merges", merge_pointers.data(), merge_pointers.size());
        gguf_set_arr_data(g, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), types.size());
        require(gguf_write_to_file(g, path.c_str(), true), "write tokenizer fixture");
        gguf_free(g);
    }
    const Tokenizer tokenizer(path, int(tokens.size()));
    using T = std::vector<int32_t>;
    require(!tokenizer.byte_fallback() && tokenizer.size() == int(tokens.size()), "BPE tokenizer");
    require(tokenizer.metadata().name == "unit" && tokenizer.metadata().temperature == 0.5f && !tokenizer.metadata().top_k, "metadata");
    require(tokenizer.encode("hello world", true) == T({hello, space_w, 'o', 'r', 'l', 'd'}), "merges apply by priority");
    require(tokenizer.encode("lll", true) == T({ll, 'l'}), "overlapping merges go left to right");
    require(tokenizer.encode("hello<think>hello<|im_end|>", true) == T({hello, think, hello, end}), "added tokens");
    require(tokenizer.encode("<|im_end|>", false).size() == 10 && tokenizer.encode("<think>", false) == T({think}),
            "control tokens are text without parse_special; user-defined tokens always match");
    require(tokenizer.encode("[PAD]", true).size() == 5, "unused tokens are not matched");
    require(tokenizer.encode("", true).empty(), "empty text");
    const std::string mixed = "hello 한글\n\t\xF0\x9F\x98\x80 it's 12";
    require(tokenizer.decode(tokenizer.encode(mixed, true), false) == mixed, "round trip");
    require(tokenizer.decode({hello, end, think, unused}, false) == "hello<think>", "control and unused tokens are hidden");
    require(tokenizer.decode({hello, end, think}, true) == "hello<|im_end|><think>", "special rendering");
    require(tokenizer.piece(0xED, false) == "\xED" && tokenizer.decode({0xED, 0x95}, false) == "\xEF\xBF\xBD", "partial characters");
    require(tokenizer.is_end(end) && !tokenizer.is_end(think) && tokenizer.is_control(end) && !tokenizer.is_control(think), "token classes");
    require(tokenizer.added_token("<think>") == think && tokenizer.added_token("nope") == -1 && tokenizer.eos() == end, "lookup");
    rejects<std::out_of_range>([&] { tokenizer.piece(int32_t(tokens.size()), true); }, "token outside vocabulary");
    rejects<std::runtime_error>([&] { Tokenizer(path, 7); }, "vocabulary size mismatch");

    // A GGUF without a tokenizer falls back to bytes.
    const auto bare = (directory / "bare.gguf").string();
    {
        gguf_context * g = gguf_init_empty();
        gguf_set_val_str(g, "general.name", "bare");
        require(gguf_write_to_file(g, bare.c_str(), true), "write bare fixture");
        gguf_free(g);
    }
    const Tokenizer bytes(bare, 256);
    require(bytes.byte_fallback() && bytes.encode("a\xED\x95\x9C", true) == T({'a', 0xED, 0x95, 0x9C}), "byte tokenizer");
    require(bytes.decode({'a', 0xED, 0x95, 0x9C}, false) == "a\xED\x95\x9C" && !bytes.is_end(0) && bytes.eos() == -1, "byte decode");
    require(Tokenizer(bare, 100).encode("\xFF", true) == T({255 % 100}), "bytes wrap into a small vocabulary");
}

static const char * const qwen38_marker = "<|im_start|> <function= reasoning_effort";
static const char * const qwen35_marker = "<|im_start|> <function= enable_thinking";

static void test_render() {
    require(ChatFormat(qwen38_marker, "auto").kind() == ChatFormat::Kind::Qwen38, "detect qwen38");
    require(ChatFormat(qwen35_marker, "auto").kind() == ChatFormat::Kind::Qwen35, "detect qwen35");
    require(ChatFormat("", "auto").kind() == ChatFormat::Kind::ChatML && ChatFormat("{{ messages }}", "auto").kind() == ChatFormat::Kind::ChatML, "fallback");
    require(ChatFormat("", "qwen38").kind() == ChatFormat::Kind::Qwen38, "explicit format");
    rejects<std::invalid_argument>([] { ChatFormat("", "llama2"); }, "unknown format name");
    const ChatFormat v38("", "qwen38"), v35("", "qwen35"), chatml("", "chatml");
    const std::vector<Message> hi = {{"user", " Hi ", std::nullopt, {}}};
    ChatOptions off; off.enable_thinking = false;
    ChatOptions medium; medium.reasoning_effort = "medium";

    auto prompt = v38.render(hi, off);
    require(prompt.text == "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n" && !prompt.thinking, "qwen38 without thinking");
    prompt = v38.render(hi, medium);
    require(prompt.text == "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n" && prompt.thinking, "qwen38 thinks by default");
    prompt = v38.render(hi, {});
    require(prompt.text.rfind("<|im_start|>system\nReasoning effort is set to xhigh.", 0) == 0, "qwen38 default effort instructions");
    ChatOptions bad_effort; bad_effort.reasoning_effort = "extreme";
    rejects<ChatError>([&] { v38.render(hi, bad_effort); }, "unknown effort");
    prompt = v35.render(hi, {});
    require(prompt.text == "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n" && !prompt.thinking, "qwen35 does not think by default");
    ChatOptions on; on.enable_thinking = true;
    require(v35.render(hi, on).thinking && v35.render(hi, on).text.ends_with("assistant\n<think>\n"), "qwen35 with thinking");
    require(chatml.render(hi, on).text == "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n" && !chatml.render(hi, on).thinking, "ChatML");

    // A tool round trip: schema, call with typed arguments, grouped responses.
    ToolCall call; call.name = "get_weather";
    call.arguments = Json::parse(R"({"city":"Seoul","days":3,"deep":{"a":[1,true,null]},"flag":false})");
    const std::vector<Message> tools_turn = {
        {"system", "sys", std::nullopt, {}}, {"user", "weather?", std::nullopt, {}},
        {"assistant", "Checking.", std::string("need tool"), {call}},
        {"tool", "sunny", std::nullopt, {}}, {"tool", "rain", std::nullopt, {}}};
    ChatOptions with_tools = off;
    with_tools.tools = Json::parse(R"([{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}])");
    prompt = v38.render(tools_turn, with_tools);
    const std::string expected_tail =
        "</IMPORTANT>\n\nsys<|im_end|>\n<|im_start|>user\nweather?<|im_end|>\n"
        "<|im_start|>assistant\n<think>\nneed tool\n</think>\n\nChecking.\n\n<tool_call>\n<function=get_weather>\n"
        "<parameter=city>\nSeoul\n</parameter>\n<parameter=days>\n3\n</parameter>\n"
        "<parameter=deep>\n{\"a\": [1, true, null]}\n</parameter>\n<parameter=flag>\nfalse\n</parameter>\n"
        "</function>\n</tool_call><|im_end|>\n"
        "<|im_start|>user\n<tool_response>\nsunny\n</tool_response>\n<tool_response>\nrain\n</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n";
    require(prompt.text.ends_with(expected_tail), "qwen38 tool turn");
    require(prompt.text.rfind("<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n"
            "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"parameters\": {\"type\": \"object\", "
            "\"properties\": {\"city\": {\"type\": \"string\"}}}}}\n</tools>\n\nIf you choose", 0) == 0, "tool schema rendering");
    // The older template writes scalars as Python does.
    require(v35.render(tools_turn, with_tools).text.find("<parameter=flag>\nFalse\n</parameter>") != std::string::npos, "qwen35 scalar arguments");

    // Reasoning of turns before the last query is kept only when preserved.
    const std::vector<Message> history = {
        {"user", "q1", std::nullopt, {}}, {"assistant", "a1", std::string("t1"), {}}, {"user", "q2", std::nullopt, {}}};
    require(v38.render(history, off).text.find("<think>\nt1\n</think>\n\na1") != std::string::npos, "qwen38 preserves thinking");
    ChatOptions drop = off; drop.preserve_thinking = false;
    require(v38.render(history, drop).text.find("assistant\na1<|im_end|>") != std::string::npos, "qwen38 can drop thinking");
    require(v35.render(history, off).text.find("assistant\na1<|im_end|>") != std::string::npos, "qwen35 drops earlier thinking");
    const std::vector<Message> inline_thought = {
        {"user", "q", std::nullopt, {}}, {"assistant", "<think>\nhmm\n</think>\n\nanswer", std::nullopt, {}}, {"user", "q2", std::nullopt, {}}};
    require(v35.render(inline_thought, off).text.find("assistant\nanswer<|im_end|>") != std::string::npos, "qwen35 strips inline thinking");

    // A trailing assistant message is continued; a forced tool call opens the reply.
    const std::vector<Message> resume = {{"user", "Hi", std::nullopt, {}}, {"assistant", "{\"a\":", std::nullopt, {}}};
    require(v38.render(resume, {}).text == "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{\"a\":", "assistant prefill");
    ChatOptions forced = with_tools; forced.forced_tool = "get_weather"; forced.enable_thinking = true;
    prompt = v38.render(hi, forced);
    require(prompt.reply_prefix == "<tool_call>\n<function=get_weather>\n" && prompt.text.ends_with("</think>\n\n" + prompt.reply_prefix) && !prompt.thinking, "forced tool");
    forced.forced_tool = "*";
    require(v38.render(hi, forced).reply_prefix == "<tool_call>\n<function=", "any tool");

    rejects<ChatError>([&] { v38.render({}, off); }, "no messages");
    rejects<ChatError>([&] { v38.render({{"system", "s", std::nullopt, {}}}, off); }, "no user query");
    rejects<ChatError>([&] { v38.render({{"user", "u", std::nullopt, {}}, {"system", "s", std::nullopt, {}}}, off); }, "late system message");
    rejects<ChatError>([&] { v38.render({{"user", "u", std::nullopt, {}}, {"narrator", "s", std::nullopt, {}}}, off); }, "unknown role");
}

struct Parsed {
    std::string reasoning, content;
    std::vector<ToolCall> calls;
    bool operator==(const Parsed & other) const {
        if (reasoning != other.reasoning || content != other.content || calls.size() != other.calls.size()) return false;
        for (size_t i = 0; i < calls.size(); ++i)
            if (calls[i].name != other.calls[i].name || calls[i].arguments.dump() != other.calls[i].arguments.dump()) return false;
        return true;
    }
};
// Parses `text` delivered in pieces of `step` bytes.
static Parsed parse(const std::string & text, size_t step, bool thinking, const Json * tools, bool inline_reasoning = false) {
    ChatParser parser(thinking, tools, inline_reasoning);
    Parsed out;
    const auto add = [&](const ChatDelta & delta) {
        out.reasoning += delta.reasoning; out.content += delta.content;
        out.calls.insert(out.calls.end(), delta.tool_calls.begin(), delta.tool_calls.end());
    };
    for (size_t at = 0; at < text.size(); at += step) add(parser.feed(text.substr(at, step)));
    add(parser.finish());
    return out;
}
// The same result regardless of how the text is cut.
static Parsed parse_all(const std::string & text, bool thinking, const Json * tools, bool inline_reasoning = false) {
    const Parsed whole = parse(text, text.size() + 1, thinking, tools, inline_reasoning);
    for (size_t step : {size_t(1), size_t(2), size_t(3), size_t(7)})
        require(parse(text, step, thinking, tools, inline_reasoning) == whole, "parsing depends on chunking: " + text);
    return whole;
}

static void test_parser() {
    const Json tools = Json::parse(R"([{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{
        "s":{"type":"string"},"n":{"type":"integer"},"o":{"type":"object"}}}}}])");
    auto r = parse_all("step 1\nstep 2\n</think>\n\nThe answer.", true, nullptr);
    require(r.reasoning == "step 1\nstep 2" && r.content == "The answer." && r.calls.empty(), "thinking then content");
    r = parse_all("plain <b>text</b> with <tool_call> kept\n", false, nullptr);
    require(r.content == "plain <b>text</b> with <tool_call> kept\n", "no tools: text passes through");
    r = parse_all("<think>\nhm\n</think>\n\nok", false, nullptr);
    require(r.reasoning == "hm" && r.content == "ok", "unprompted thinking section");
    r = parse_all("never closed", true, nullptr);
    require(r.reasoning == "never closed" && r.content.empty(), "unterminated thinking");
    r = parse_all("a\n</think>\n\nb", true, nullptr, true);
    require(r.content == "<think>\na\n</think>\n\nb" && r.reasoning.empty(), "inline reasoning");

    r = parse_all("Let me look.\n\n<tool_call>\n<function=f>\n<parameter=s>\n42\n</parameter>\n<parameter=n>\n42\n</parameter>\n"
                  "<parameter=o>\n{\"k\": [1, 2]}\n</parameter>\n<parameter=free>\nline 1\nline 2\n</parameter>\n</function>\n</tool_call>", false, &tools);
    require(r.content == "Let me look." && r.calls.size() == 1 && r.calls[0].name == "f", "tool call");
    require(r.calls[0].arguments.dump() == R"({"s":"42","n":42,"o":{"k":[1,2]},"free":"line 1\nline 2"})", "arguments are typed by the schema");
    r = parse_all("<tool_call>\n<function=f>\n</function>\n</tool_call>\n<tool_call>\n{\"name\": \"g\", \"arguments\": {\"x\": 1}}\n</tool_call>", false, &tools);
    require(r.calls.size() == 2 && r.calls[0].arguments.dump() == "{}" && r.calls[1].name == "g" && r.calls[1].arguments.dump() == "{\"x\":1}" && r.content.empty(),
            "several calls, including the JSON form");
    r = parse_all("a <tool> b <tool_cal", false, &tools);
    require(r.content == "a <tool> b <tool_cal" && r.calls.empty(), "look-alike tags are content");
    r = parse_all("x<tool_call>\n<function=f>\n<parameter=n>\n7\n</parameter>", false, &tools);
    require(r.calls.size() == 1 && r.calls[0].arguments.dump() == "{\"n\":7}" && r.content == "x", "unterminated call is still parsed");
    r = parse_all("x <tool_call>garbage</tool_call> y", false, &tools);
    require(r.content == "x <tool_call>garbage</tool_call> y" && r.calls.empty(), "unparsable call stays text");
    r = parse_all("t\n</think>\n\n<tool_call>\n<function=f>\n<parameter=s>\nv\n</parameter>\n</function>\n</tool_call>", true, &tools);
    require(r.reasoning == "t" && r.content.empty() && r.calls.size() == 1, "thinking then tool call");
    // The opening of a forced call is fed before the generated text.
    ChatParser forced(false, &tools, false);
    ChatDelta delta = forced.feed("<tool_call>\n<function=");
    require(delta.empty(), "forced call opening is not output");
    delta = forced.feed("f>\n<parameter=n>\n1\n</parameter>\n</function>\n</tool_call>");
    require(delta.tool_calls.size() == 1 && delta.tool_calls[0].name == "f" && forced.finish().empty(), "forced call");
}

int main() {
    const auto directory = std::filesystem::temp_directory_path() / ("tidal-server-test-" + std::to_string(::getpid()));
    try {
        std::filesystem::create_directories(directory);
        test_json();
        test_split();
        test_tokenizer(directory);
        test_render();
        test_parser();
        std::filesystem::remove_all(directory);
        std::cout << "server text tests passed\n";
        return 0;
    } catch (const std::exception & e) {
        std::error_code ignored;
        std::filesystem::remove_all(directory, ignored);
        std::cerr << "server test failed: " << e.what() << '\n';
        return 1;
    }
}
