#pragma once
#include "internal.hpp"
#include "profiler.hpp"
#include <deque>
#include <mutex>

namespace tidal {
// Source lifetimes stay on the calling thread: releasing a completed item may
// release GPU workspace accounted by the prefill memory planner.
class KVExporter {
    struct Pending {
        std::shared_ptr<void> source_owner;
        Buffer staging{nullptr, ggml_backend_buffer_free};
        std::future<double> work;
    };
    Backend transfer_{nullptr, ggml_backend_free};
    ggml_backend_buffer_type_t host_type_;
    Statistics & stats_;
    Profiler & profiler_;
    std::deque<Pending> pending_;
    std::mutex stream_mutex_;
    void finish_front();
public:
    KVExporter(int device, Statistics &, Profiler &);
    ~KVExporter();
    void enqueue(std::shared_ptr<void> owner, ggml_tensor * keys, ggml_tensor * values,
                 SparseAttention & destination, int start, int count, int layer);
    void reap();
    void finish();
};
} // namespace tidal
