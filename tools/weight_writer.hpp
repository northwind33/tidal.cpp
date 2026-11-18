#pragma once
#include "internal.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>

struct TypeRule { std::string pattern; ggml_type type; size_t matches = 0; };
inline bool glob_match(const std::string & pattern, const std::string & name) {
    size_t p = 0, n = 0, star = std::string::npos, retry = 0;
    while (n < name.size()) {
        if (p < pattern.size() && (pattern[p] == '?' || pattern[p] == name[n])) { ++p; ++n; }
        else if (p < pattern.size() && pattern[p] == '*') { star = p++; retry = n; }
        else if (star != std::string::npos) { p = star + 1; n = ++retry; }
        else return false;
    }
    while (p < pattern.size() && pattern[p] == '*') ++p;
    return p == pattern.size();
}
inline std::vector<TypeRule> read_type_rules(const std::string & path) {
    std::vector<TypeRule> rules;
    if (path.empty()) return rules;
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot read tensor type rules: " + path);
    auto trim = [](const std::string & s) {
        const auto begin = s.find_first_not_of(" \t\r");
        return begin == std::string::npos ? std::string{} : s.substr(begin, s.find_last_not_of(" \t\r") - begin + 1);
    };
    for (std::string line; std::getline(input, line);) {
        line = trim(line.substr(0, line.find('#')));
        if (line.empty()) continue;
        auto at = line.find('=');
        if (at == std::string::npos || trim(line.substr(0, at)).empty()) throw std::invalid_argument("Expected tensor_glob=type: " + line);
        rules.push_back({trim(line.substr(0, at)), tidal::parse_storage_type(trim(line.substr(at + 1)))});
    }
    return rules;
}

struct PlannedWeight { tidal::TensorSpec spec; ggml_tensor * tensor; };
inline std::vector<PlannedWeight> plan_weights(const tidal::Config & c, ggml_context * ctx, gguf_context * file,
                                               ggml_type matrices, ggml_type ple_type, std::vector<TypeRule> & rules) {
    std::vector<PlannedWeight> plan;
    std::map<std::string, uint64_t> by_type;
    uint64_t parameters = 0, bytes = 0;
    for (const auto & spec : tidal::tensor_specs(c)) {
        auto type = (spec.kind == tidal::TensorSpec::Matrix || spec.kind == tidal::TensorSpec::Expert) ? matrices :
            spec.kind == tidal::TensorSpec::Embedding ? GGML_TYPE_BF16 :
            spec.kind == tidal::TensorSpec::Lookup ? ple_type : GGML_TYPE_F32;
        for (auto & rule : rules) if (glob_match(rule.pattern, spec.name)) { ++rule.matches; type = rule.type; }
        if (spec.columns % ggml_blck_size(type))
            throw std::invalid_argument(spec.name + ": row width " + std::to_string(spec.columns) +
                " is not divisible by " + std::to_string(ggml_blck_size(type)) + " for " + ggml_type_name(type));
        auto * tensor = ggml_new_tensor_3d(ctx, type, spec.columns, spec.rows, spec.slices);
        ggml_set_name(tensor, spec.name.c_str());
        gguf_add_tensor(file, tensor);
        plan.push_back({spec, tensor});
        const uint64_t count = uint64_t(ggml_nelements(tensor));
        parameters += count; bytes += ggml_nbytes(tensor); by_type[ggml_type_name(type)] += count;
    }
    for (const auto & rule : rules) if (!rule.matches) throw std::invalid_argument("Tensor rule matched nothing: " + rule.pattern);
    std::cout << "parameters=" << parameters << " tensor_bytes=" << bytes << " effective_bpw="
              << std::setprecision(9) << double(bytes) * 8 / double(parameters) << '\n';
    for (const auto & [type, count] : by_type) std::cout << "  " << type << ": " << count << " parameters\n";
    return plan;
}

// Bounded-memory, row-aligned generation. Fill is called in tensor/element order
// independent of quantization threads and chunk size, preserving seeded weights.
template<class Fill>
void write_weights(const std::string & path, gguf_context * file, const std::vector<PlannedWeight> & plan,
                   int chunk_mib, int threads, Fill fill) {
    const auto target = std::filesystem::path(path);
    const auto temporary = std::filesystem::path(path + ".partial");
    if (std::filesystem::exists(target) || std::filesystem::exists(temporary))
        throw std::invalid_argument("Refusing to overwrite model or partial output: " + path);
    const uint64_t meta = gguf_get_meta_size(file);
    const auto alignment = gguf_get_alignment(file);
    auto aligned = [&](uint64_t n) { return (n + alignment - 1) / alignment * alignment; };
    uint64_t total = meta;
    for (const auto & w : plan) total += aligned(ggml_nbytes(w.tensor));
    auto parent = target.parent_path();
    if (parent.empty()) parent = ".";
    if (std::filesystem::space(parent).available < total) throw std::runtime_error("Insufficient disk space for GGUF");
    bool owned = false;
    try {
        FILE * metadata = std::fopen(temporary.string().c_str(), "wbx");
        if (!metadata) throw std::runtime_error("Cannot exclusively create partial GGUF");
        owned = true;
        const bool meta_ok = gguf_write_to_file_ptr(file, metadata, true);
        const int close_status = std::fclose(metadata);
        if (!meta_ok || close_status) throw std::runtime_error("Cannot write GGUF metadata");
        std::ofstream output(temporary, std::ios::binary | std::ios::app);
        if (!output) throw std::runtime_error("Cannot open GGUF payload output");
        uint64_t written = meta;
        for (const auto & w : plan) {
            auto * tensor = w.tensor;
            const int64_t width = w.spec.columns, rows = w.spec.rows * w.spec.slices;
            const int64_t chunk_rows = std::max<int64_t>(1, int64_t(chunk_mib) * 1024 * 1024 / (width * sizeof(float)));
            const size_t row_bytes = ggml_row_size(tensor->type, width);
            std::vector<float> source(size_t(std::min(rows, chunk_rows) * width));
            std::vector<uint8_t> encoded(size_t(std::min(rows, chunk_rows)) * row_bytes);
            // Synthetic dummy calibration only. This is not a trained-model imatrix.
            std::vector<float> importance(ggml_quantize_requires_imatrix(tensor->type) ? size_t(width) : 0, 1.0f);
            ggml_quantize_init(tensor->type);
            for (int64_t first = 0; first < rows; first += chunk_rows) {
                const int64_t count = std::min(chunk_rows, rows - first);
                fill(w.spec, source.data(), count * width);
                const int workers = int(std::min<int64_t>(threads, count));
                auto encode = [&](int worker) {
                    const int64_t begin = count * worker / workers, end = count * (worker + 1) / workers;
                    ggml_quantize_chunk(tensor->type, source.data() + begin * width, encoded.data() + begin * row_bytes,
                                        0, end - begin, width, importance.empty() ? nullptr : importance.data());
                };
                std::vector<std::jthread> pool;
                for (int worker = 1; worker < workers; ++worker) pool.emplace_back(encode, worker);
                encode(0);
                pool.clear();
                const size_t bytes = size_t(count) * row_bytes;
                if (!ggml_validate_row_data(tensor->type, encoded.data(), bytes)) throw std::runtime_error("Invalid generated weight: " + w.spec.name);
                output.write(reinterpret_cast<const char *>(encoded.data()), std::streamsize(bytes));
                if (!output) throw std::runtime_error("GGUF payload write failed");
                written += bytes;
            }
            const size_t padding = size_t(aligned(ggml_nbytes(tensor)) - ggml_nbytes(tensor));
            std::vector<char> zeros(padding, 0);
            output.write(zeros.data(), std::streamsize(padding));
            written += padding;
            if (total > 1024 * 1024 * 1024ULL)
                std::cerr << w.spec.name << ": " << written << '/' << total << " bytes\n";
        }
        output.close();
        if (!output || written != total || std::filesystem::file_size(temporary) != total)
            throw std::runtime_error("Incomplete GGUF output");
        // Hard-link publication fails if another process created target meanwhile.
        std::filesystem::create_hard_link(temporary, target);
        std::filesystem::remove(temporary);
        owned = false;
        std::cout << path << ": " << total << " GGUF bytes written\n";
    } catch (...) {
        if (owned) { std::error_code error; std::filesystem::remove(temporary, error); }
        throw;
    }
}
