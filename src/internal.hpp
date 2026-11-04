#pragma once

#include "tidal/tidal.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "gguf.h"
#include <map>
#include <stdexcept>
#include <utility>

namespace tidal {
constexpr const char * architecture = "tidal-qwen4exp-dense";
constexpr uint32_t format_version = 1;

using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using Gguf = std::unique_ptr<gguf_context, decltype(&gguf_free)>;
using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
using Backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;

struct TensorSpec {
    std::string name;
    int64_t columns;
    int64_t rows;
    enum Kind { Matrix, Norm, Conv, ALog, Bias, Embedding } kind;
};
std::vector<TensorSpec> tensor_specs(const Config & c);
void write_config(gguf_context * file, const Config & config);
Config read_config(const gguf_context * file);

struct Model {
    Config config;
    Context ctx{nullptr, ggml_free};
    Buffer buffer{nullptr, ggml_backend_buffer_free};
    std::map<std::string, ggml_tensor *> weights;
    Model(const std::string & path, ggml_backend_t backend);
    ggml_tensor * at(const std::string & name) const { return weights.at(name); }
};

class SparseAttention {
public:
    SparseAttention(int heads, int kv_heads, int dim, int capacity, int threads);
    void append(const std::vector<float> & key, const std::vector<float> & value);
    std::vector<float> attend(const std::vector<float> & query,
                              const std::vector<int32_t> & selected) const;
    void reset() { size_ = 0; }
    int size() const { return size_; }
private:
    int heads_, kv_heads_, dim_, capacity_, threads_, size_ = 0;
    std::vector<ggml_bf16_t> keys_, values_;
};
std::vector<int32_t> expand_blocks(const std::vector<int32_t> & blocks,
                                  int visible, int block_size);
} // namespace tidal
