#pragma once
#include <charconv>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

inline int number(const std::string & value) {
    int n = 0;
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), n);
    if (error != std::errc{} || end != value.data() + value.size())
        throw std::invalid_argument("Invalid integer: " + value);
    return n;
}
class Args {
    std::map<std::string, std::string> values_;
public:
    Args(int argc, char ** argv, const std::vector<std::string> & allowed) {
        for (int i = 1; i < argc; ++i) {
            std::string name = argv[i];
            bool known = false;
            for (const auto & a : allowed) if (a == name) known = true;
            if (!known || values_.contains(name)) throw std::invalid_argument("Unknown or repeated option: " + name);
            if (name == "--help") { values_[name] = "1"; continue; }
            if (++i == argc) throw std::invalid_argument("Missing value for " + name);
            values_[name] = argv[i];
        }
    }
    std::string get(const std::string & name, std::string fallback = "") const {
        auto it = values_.find(name);
        return it == values_.end() ? std::move(fallback) : it->second;
    }
    int integer(const std::string & name, int fallback) const {
        return number(get(name, std::to_string(fallback)));
    }
};
