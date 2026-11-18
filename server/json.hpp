#pragma once
// Minimal JSON value for the HTTP server: ordered objects (key order is part of
// rendered tool schemas), 64-bit integers kept apart from doubles, strict
// parsing and output that is always valid UTF-8.
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace tidal::server {

struct JsonError : std::runtime_error { using std::runtime_error::runtime_error; };

class Json {
public:
    using Array = std::vector<Json>;
    using Member = std::pair<std::string, Json>;
    using Object = std::vector<Member>;

    Json() = default;
    Json(std::nullptr_t) {}
    Json(bool value) : value_(value) {}
    Json(int value) : value_(int64_t(value)) {}
    Json(long value) : value_(int64_t(value)) {}
    Json(long long value) : value_(int64_t(value)) {}
    Json(unsigned value) : value_(int64_t(value)) {}
    Json(unsigned long value) : value_(int64_t(value)) {}
    Json(unsigned long long value) : value_(int64_t(value)) {}
    Json(double value) : value_(value) {}
    // The nearest double of the float's shortest decimal form: 0.8f is written as 0.8.
    Json(float value);
    Json(const char * value) : value_(std::string(value)) {}
    Json(std::string value) : value_(std::move(value)) {}
    Json(std::string_view value) : value_(std::string(value)) {}
    Json(Array value) : value_(std::move(value)) {}
    Json(Object value) : value_(std::move(value)) {}
    static Json array() { return Json(Array{}); }
    static Json object() { return Json(Object{}); }
    static Json object(std::initializer_list<Member> members) { return Json(Object(members)); }

    bool is_null() const { return std::holds_alternative<std::monostate>(value_); }
    bool is_bool() const { return std::holds_alternative<bool>(value_); }
    bool is_int() const { return std::holds_alternative<int64_t>(value_); }
    bool is_number() const { return is_int() || std::holds_alternative<double>(value_); }
    bool is_string() const { return std::holds_alternative<std::string>(value_); }
    bool is_array() const { return std::holds_alternative<Array>(value_); }
    bool is_object() const { return std::holds_alternative<Object>(value_); }

    bool boolean() const { return std::get<bool>(value_); }
    int64_t integer() const { return std::get<int64_t>(value_); }
    double number() const { return is_int() ? double(std::get<int64_t>(value_)) : std::get<double>(value_); }
    const std::string & string() const { return std::get<std::string>(value_); }
    const Array & items() const { return std::get<Array>(value_); }
    Array & items() { return std::get<Array>(value_); }
    const Object & members() const { return std::get<Object>(value_); }
    Object & members() { return std::get<Object>(value_); }

    // Null when this is not an object or the key is absent.
    const Json * find(std::string_view key) const;
    // Shared null value for absent keys.
    const Json & at(std::string_view key) const;
    bool contains(std::string_view key) const { return find(key) != nullptr; }
    // Object access that inserts; a null value becomes an object.
    Json & operator[](std::string_view key);
    void push(Json value);
    size_t size() const { return is_array() ? items().size() : is_object() ? members().size() : 0; }

    // `python` adds the ", " and ": " separators of Python's json.dumps, which is
    // what the model's chat template renders tool schemas and arguments with.
    std::string dump(bool python = false) const;
    static Json parse(std::string_view text);

private:
    std::variant<std::monostate, bool, int64_t, double, std::string, Array, Object> value_;
    void write(std::string & out, bool python) const;
};

// Replaces each ill-formed sequence with one U+FFFD (as text decoders do); the
// result is valid UTF-8.
std::string sanitize_utf8(std::string_view text);
// Length of the longest prefix that does not end inside a multi-byte sequence.
size_t utf8_complete_prefix(std::string_view text);
void append_utf8(std::string & out, uint32_t codepoint);

} // namespace tidal::server
