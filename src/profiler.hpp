#pragma once
#include "tidal/tidal.hpp"
#include <chrono>
#include <mutex>
#include <stdexcept>
#include <string_view>
#ifdef TIDAL_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

namespace tidal {
class Profiler {
    using Clock = std::chrono::steady_clock;
    Clock::time_point origin_ = Clock::now();
    std::mutex mutex_;
public:
    ProfileMode mode = ProfileMode::Off;
    bool nvtx = false;
    uint64_t event_limit, dropped = 0;
    std::vector<ProfileEvent> events;
    Profiler(ProfileMode m, bool n, uint64_t limit) : mode(m), nvtx(n), event_limit(limit) {
        if (!limit) throw std::invalid_argument("Profile event limit must be positive");
#ifndef TIDAL_NVTX
        if (nvtx) throw std::invalid_argument("NVTX is unavailable in this build");
#endif
    }
    bool enabled() const { return mode != ProfileMode::Off || nvtx; }
    double now() const { return std::chrono::duration<double, std::micro>(Clock::now() - origin_).count(); }
    void clear() { events.clear(); dropped = 0; origin_ = Clock::now(); }
    void record(ProfileEvent event) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (events.size() < event_limit) events.push_back(std::move(event));
        else ++dropped;
    }
};

class ProfileScope {
    Profiler * profiler_ = nullptr;
    double start_ = 0;
public:
    ProfileEvent event;
    ProfileScope(Profiler * p, std::string_view category, std::string_view name,
                 std::string_view phase, int position, int layer, std::string_view backend,
                 int head = -1) {
        if (!p || !p->enabled()) return;
        profiler_ = p;
        event.category = category; event.name = name; event.phase = phase;
        event.position = position; event.layer = layer; event.backend = backend; event.head = head;
#ifdef TIDAL_NVTX
        if (p->nvtx) {
            const auto label = std::string(phase) + "/layer=" + std::to_string(layer) +
                "/token=" + std::to_string(position) + "/" + std::string(name);
            nvtxRangePushA(label.c_str());
        }
#endif
        start_ = p->now();
    }
    ProfileScope(const ProfileScope &) = delete;
    ProfileScope & operator=(const ProfileScope &) = delete;
    void restart() { if (profiler_) start_ = profiler_->now(); }
    void finish() {
        if (!profiler_) return;
        event.start_us = start_;
        event.wall_us = profiler_->now() - start_;
#ifdef TIDAL_NVTX
        if (profiler_->nvtx) nvtxRangePop();
#endif
        auto * p = profiler_;
        profiler_ = nullptr;
        if (p->mode != ProfileMode::Off) p->record(std::move(event));
    }
    ~ProfileScope() { finish(); }
};
} // namespace tidal
