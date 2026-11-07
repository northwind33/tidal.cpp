#include "kv_export.hpp"
#ifdef TIDAL_CUDA
#include "ggml-cuda.h"
#endif
#include <algorithm>

namespace tidal {
KVExporter::KVExporter(int device, Statistics & stats, Profiler & profiler)
    : host_type_(nullptr), stats_(stats), profiler_(profiler) {
#ifdef TIDAL_CUDA
    transfer_.reset(ggml_backend_cuda_init(device));
    host_type_ = ggml_backend_cuda_host_buffer_type();
#else
    (void) device;
#endif
    if (!transfer_ || !host_type_) throw std::runtime_error("GPU KV export requires a CUDA build and device");
}
KVExporter::~KVExporter() {
    // A failure must not destroy a source buffer while its transfer is in flight.
    for (auto & p : pending_) if (p.work.valid()) p.work.wait();
}
void KVExporter::finish_front() {
    auto & item = pending_.front();
    try {
        const double ms = item.work.get();
        stats_.transfer_ms += ms;
    } catch (...) { pending_.pop_front(); throw; }
    pending_.pop_front();
}
void KVExporter::reap() {
    while (!pending_.empty() && pending_.front().work.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
        finish_front();
}
void KVExporter::finish() { while (!pending_.empty()) finish_front(); }
void KVExporter::enqueue(std::shared_ptr<void> owner, ggml_tensor * keys, ggml_tensor * values,
                         SparseAttention & destination, int start, int count, int layer) {
    if (count <= 0 || start != destination.size() || keys->type != GGML_TYPE_BF16 || values->type != GGML_TYPE_BF16 ||
        !ggml_is_contiguous(keys) || !ggml_is_contiguous(values) || !ggml_are_same_shape(keys, values) ||
        keys->ne[1] < start + count || keys->ne[2] != 1 || keys->ne[3] != 1)
        throw std::invalid_argument("Invalid contiguous BF16 KV export range");
    reap();
    // Bound both pinned memory and outstanding source buffers. Attention and FFN
    // can still overlap the current layer's transfer on a separate stream.
    if (pending_.size() >= 2) finish_front();
    const size_t row_bytes = size_t(keys->ne[0]) * sizeof(ggml_bf16_t);
    const int chunk_rows = int(std::min<size_t>(size_t(count), std::max<size_t>(1, (32ull * 1024 * 1024) / row_bytes)));
    const size_t half_bytes = size_t(chunk_rows) * row_bytes;
    Pending item;
    item.source_owner = std::move(owner);
    item.staging.reset(ggml_backend_buft_alloc_buffer(host_type_, 2 * half_bytes));
    if (!item.staging || ggml_backend_buffer_get_type(item.staging.get()) != host_type_)
        throw std::runtime_error("Pinned host memory is required for asynchronous KV export");
    auto * host = static_cast<char *>(ggml_backend_buffer_get_base(item.staging.get()));
    ProfileScope enqueue(&profiler_, "host", "kv_export_enqueue", "prefill.kv", start, layer, "CPU");
    item.work = std::async(std::launch::async, [this, keys, values, &destination, start, count, layer, host, row_bytes, chunk_rows, half_bytes] {
        std::lock_guard<std::mutex> lock(stream_mutex_);
        double transfer_ms = 0;
        try {
            for (int offset = 0; offset < count; offset += chunk_rows) {
                const int rows = std::min(chunk_rows, count - offset);
                const size_t bytes = size_t(rows) * row_bytes, source_offset = size_t(start + offset) * row_bytes;
                ProfileScope copy(&profiler_, "transfer", "D2H_async", "prefill.kv", start + offset, layer, ggml_backend_name(transfer_.get()));
                copy.event.bytes = 2 * bytes;
                const auto tick = std::chrono::steady_clock::now();
                ggml_backend_tensor_get_async(transfer_.get(), keys, host, source_offset, bytes);
                ggml_backend_tensor_get_async(transfer_.get(), values, host + half_bytes, source_offset, bytes);
                ggml_backend_synchronize(transfer_.get());
                transfer_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tick).count();
                copy.finish();
                ProfileScope commit(&profiler_, "host", "kv_commit_bf16", "prefill.kv", start + offset, layer, "CPU");
                destination.append_bf16(reinterpret_cast<const ggml_bf16_t *>(host),
                                        reinterpret_cast<const ggml_bf16_t *>(host + half_bytes), rows);
            }
        } catch (...) { ggml_backend_synchronize(transfer_.get()); throw; }
        return transfer_ms;
    });
    pending_.push_back(std::move(item));
    stats_.device_to_host_bytes += 2 * size_t(count) * row_bytes;
    stats_.prefill_kv_to_host_bytes += 2 * size_t(count) * row_bytes;
}
} // namespace tidal
