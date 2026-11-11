#pragma once
#include "prefill_memory.hpp"

namespace tidal {
class PrefillTensorStorage {
    PrefillMemory & memory_;
    ggml_backend_t backend_;
    uint64_t allocated_ = 0;
public:
    Context context{ggml_init({1024 * 1024, nullptr, true}), ggml_free};
    Buffer buffer{nullptr, ggml_backend_buffer_free};
    PrefillTensorStorage(PrefillMemory & memory, ggml_backend_t backend) : memory_(memory), backend_(backend) {
        if (!context) throw std::runtime_error("Cannot create prefill tensor metadata");
    }
    ~PrefillTensorStorage() { buffer.reset(); if (allocated_) memory_.release(allocated_); }
    ggml_tensor * tensor(ggml_type type, int64_t a, int64_t b = 1, int64_t c = 1) {
        if (buffer) throw std::logic_error("Cannot add tensors after allocating prefill storage");
        return ggml_new_tensor_3d(context.get(), type, a, b, c);
    }
    void allocate() {
        if (buffer) throw std::logic_error("Prefill storage already allocated");
        auto buft = ggml_backend_get_default_buffer_type(backend_);
        const auto bytes = ggml_backend_alloc_ctx_tensors_from_buft_size(context.get(), buft);
        memory_.acquire(bytes); allocated_ = bytes;
        buffer.reset(ggml_backend_alloc_ctx_tensors(context.get(), backend_));
        if (!buffer) throw PrefillAllocationError("Insufficient VRAM for reserved GPU prefill storage");
        ggml_backend_buffer_clear(buffer.get(), 0);
    }
};
} // namespace tidal
