#pragma once
#include "residency.hpp"
#include "profiler.hpp"
#include <functional>
#include <limits>

namespace tidal {
// The execution order is controlled by the prefill driver. This allocator only
// chooses weight victims and overlaps an optional future upload with compute.
class PrefillMemory {
    Model & model_;
    ggml_backend_t backend_;
    Statistics & stats_;
    Profiler & profiler_;
    uint64_t limit_, reserve_, workspace_ = 0;
    int layer_ = -1;
    std::vector<int> protected_;
    bool device_;
    uint64_t available() const;
    bool make_room(uint64_t bytes, bool required);
    void upload(int id, bool asynchronous, const char * phase);
public:
    // Called before evicting weights so completed KV copies can relinquish space.
    std::function<void()> reclaim_completed;
    std::function<void()> wait_transfers;
    PrefillMemory(Model &, ggml_backend_t, Statistics &, Profiler &, const Options &);
    void protect(int layer, std::vector<int> groups);
    void acquire(uint64_t bytes);
    void release(uint64_t bytes);
    void prefetch(std::vector<int> groups);
    void restore();
    uint64_t workspace_bytes() const { return workspace_; }
};
} // namespace tidal
