#include "prefill_memory.hpp"
#include "ggml-cpu.h"

namespace tidal {
PrefillMemory::PrefillMemory(Model & m, ggml_backend_t b, Statistics & s, Profiler & p, const Options & o)
    : model_(m), backend_(b), stats_(s), profiler_(p), limit_(o.prefill_memory_limit),
      reserve_(o.prefill_memory_reserve), device_(!ggml_backend_is_cpu(b)) {}

uint64_t PrefillMemory::available() const {
    size_t free = 0, total = 0;
    ggml_backend_dev_memory(ggml_backend_get_device(backend_), &free, &total);
    uint64_t result = device_ ? uint64_t(free) : std::numeric_limits<uint64_t>::max();
    result = result > reserve_ ? result - reserve_ : 0;
    if (limit_) {
        const uint64_t used = uint64_t(model_.resident_bytes()) + workspace_;
        result = std::min(result, limit_ > used ? limit_ - used : 0);
    }
    return result;
}
bool PrefillMemory::make_room(uint64_t bytes, bool required) {
    if (reclaim_completed) reclaim_completed();
    if (available() >= bytes) return true;
    for (int id : weight_eviction_order(model_.groups, layer_, protected_)) {
        if (!model_.resident(id)) continue;
        auto & g = model_.groups[id];
        ProfileScope event(&profiler_, "memory", "weight_evict", "prefill.weights", -1, g.layer, ggml_backend_name(backend_));
        event.event.bytes = g.payload_bytes;
        model_.evict(id); ++stats_.prefill_weight_evictions;
        if (available() >= bytes) return true;
    }
    if (required && wait_transfers) {
        wait_transfers();
        if (available() >= bytes) return true;
    }
    if (required) throw std::runtime_error("Insufficient VRAM for active prefill sublayer and KV/workspace: required " +
        std::to_string(bytes) + ", available " + std::to_string(available()) + "; no CPU fallback");
    return false;
}
void PrefillMemory::upload(int id, bool asynchronous, const char * phase) {
    auto & g = model_.groups[id];
    const bool absent = !model_.resident(id);
    ProfileScope event(&profiler_, "memory", asynchronous ? "weight_prefetch" : "weight_ready", phase, -1, g.layer,
                       ggml_backend_name(backend_));
    if (absent) event.event.bytes = g.payload_bytes;
    model_.load(id, asynchronous);
    if (absent) {
        ++stats_.prefill_weight_reloads;
        if (asynchronous) ++stats_.prefill_weight_prefetches;
        stats_.prefill_weight_reload_bytes += g.payload_bytes;
        if (device_) stats_.host_to_device_bytes += g.payload_bytes;
    }
}
void PrefillMemory::protect(int layer, std::vector<int> groups) {
    layer_ = layer;
    groups.erase(std::remove(groups.begin(), groups.end(), -1), groups.end());
    protected_ = std::move(groups);
    uint64_t needed = 0;
    for (int id : protected_) if (!model_.resident(id)) needed += model_.groups[id].allocation_bytes;
    make_room(needed, true);
    for (int id : protected_) upload(id, false, "prefill.weights");
}
void PrefillMemory::acquire(uint64_t bytes) {
    if (bytes > std::numeric_limits<uint64_t>::max() - workspace_) throw std::overflow_error("Prefill workspace overflow");
    make_room(bytes, true);
    workspace_ += bytes;
    stats_.prefill_peak_workspace_bytes = std::max(stats_.prefill_peak_workspace_bytes, workspace_);
}
void PrefillMemory::release(uint64_t bytes) {
    if (bytes > workspace_) throw std::logic_error("Unbalanced prefill workspace release");
    workspace_ -= bytes;
}
void PrefillMemory::prefetch(std::vector<int> groups) {
    auto saved = protected_;
    // Protect requested resident groups too: evicting one while making room for
    // another would invalidate the missing-byte estimate and undo the prefetch.
    for (int id : groups) if (id >= 0) protected_.push_back(id);
    uint64_t needed = 0;
    for (int id : groups) if (id >= 0 && !model_.resident(id)) needed += model_.groups[id].allocation_bytes;
    try {
        if (needed && make_room(needed, false))
            for (int id : groups) if (id >= 0 && !model_.resident(id)) upload(id, true, "prefill.weights");
    } catch (...) { protected_ = std::move(saved); throw; }
    protected_ = std::move(saved);
}
void PrefillMemory::restore() {
    if (workspace_) throw std::logic_error("Restore weights only after releasing prefill workspace");
    model_.wait_all();
    // Artificial prefill limits do not constrain the restored decode layout.
    // Never evict an already restored layer to disguise a restoration OOM.
    limit_ = 0;
    for (size_t id = 0; id < model_.groups.size(); ++id) {
        if (!model_.resident(int(id))) {
            if (available() < model_.groups[id].allocation_bytes)
                throw std::runtime_error("Insufficient VRAM to restore decode weights after prefill");
            upload(int(id), false, "prefill.restore");
        }
    }
}
} // namespace tidal
