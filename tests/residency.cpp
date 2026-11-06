#include "residency.hpp"
#include "prefill_memory.hpp"
#include "ggml-cpu.h"
#ifdef TIDAL_CUDA
#include "ggml-cuda.h"
#endif
#include <iostream>
#include <set>

int main(int argc, char ** argv) {
    try {
        if (argc != 3) throw std::runtime_error("Usage: tidal-residency-test model.gguf cpu|cuda");
        // Exact policy fixture: completed layers before farthest future layers,
        // protected active attention excluded, inactive current FFN last.
        std::vector<tidal::WeightGroup> fixture(7);
        const int layers[] = {8, 0, 4, 2, 2, 6, 1};
        for (int i = 0; i < 7; ++i) fixture[i].layer = layers[i];
        const std::vector<int> protected_groups{3};
        if (tidal::weight_eviction_order(fixture, 2, protected_groups) != std::vector<int>{1,6,0,5,2,4})
            throw std::runtime_error("Incorrect residency priority");
        tidal::Backend backend(nullptr, ggml_backend_free);
        if (std::string(argv[2]) == "cpu") backend.reset(ggml_backend_cpu_init());
#ifdef TIDAL_CUDA
        else backend.reset(ggml_backend_cuda_init(0));
#endif
        if (!backend) throw std::runtime_error("Backend unavailable");
        tidal::Model model(argv[1], backend.get());
        std::map<std::string, std::vector<char>> originals;
        for (const auto & [name, tensor] : model.weights) {
            auto & bytes = originals[name]; bytes.resize(ggml_nbytes(tensor));
            ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
        }
        const size_t resident = model.resident_bytes();
        for (int i = 0; i < model.config.layers; ++i)
            for (auto part : {tidal::WeightPart::Attention, tidal::WeightPart::FeedForward})
                if (model.group(i, part) < 0) throw std::runtime_error("Missing sublayer group");
        std::set<ggml_tensor *> unique;
        for (const auto & g : model.groups) for (auto * t : g.tensors)
            if (!unique.insert(t).second) throw std::runtime_error("Aliased weight allocated twice");
        for (size_t i = 0; i < model.groups.size(); ++i) {
            model.evict(int(i));
            if (model.resident(int(i))) throw std::runtime_error("Eviction retained storage");
            for (auto * t : model.groups[i].tensors)
                if (t->data || t->buffer) throw std::runtime_error("Eviction left dangling tensor storage");
        }
        if (model.resident_bytes()) throw std::runtime_error("Weight memory not released");
        for (size_t i = model.groups.size(); i-- > 0;) model.load(int(i), true);
        model.wait_all();
        if (resident != model.resident_bytes()) throw std::runtime_error("Reload changed weight allocation");
        for (const auto & [name, tensor] : model.weights) {
            std::vector<char> bytes(ggml_nbytes(tensor));
            ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
            if (bytes != originals.at(name)) throw std::runtime_error("Reload changed tensor bytes: " + name);
        }
        if (model.config.tied_embeddings && model.at("token_embd.weight") != model.at("output.weight"))
            throw std::runtime_error("Tied embedding alias was lost");
        // Evicting a pending upload waits before invalidating storage; a later
        // synchronous demand load must still be exact.
        const int last = int(model.groups.size()) - 1;
        model.evict(last); model.load(last, true); model.evict(last); model.load(last);
        for (auto * t : model.groups[last].tensors) {
            std::vector<char> bytes(ggml_nbytes(t)); ggml_backend_tensor_get(t, bytes.data(), 0, bytes.size());
            if (bytes != originals.at(t->name)) throw std::runtime_error("Pending reload lifetime error");
        }
        tidal::Options options;
        options.prefill_memory_limit = resident;
        options.prefill_memory_reserve = 0;
        tidal::Statistics stats;
        tidal::Profiler profile(tidal::ProfileMode::Phases, false, 10000);
        tidal::PrefillMemory memory(model, backend.get(), stats, profile, options);
        const int active = model.group(2, tidal::WeightPart::Attention);
        const uint64_t workspace = resident - model.groups[active].allocation_bytes;
        memory.protect(2, {active});
        memory.acquire(workspace);
        if (!model.resident(active) || !stats.prefill_weight_evictions)
            throw std::runtime_error("Memory pressure evicted active weights or ignored the cap");
        bool seen_future = false;
        int previous_future = model.config.layers + 1;
        for (const auto & event : profile.events) if (event.name == "weight_evict") {
            if (event.layer < 2 && seen_future) throw std::runtime_error("Completed layers were not preferred");
            if (event.layer > 2) {
                seen_future = true;
                if (event.layer > previous_future) throw std::runtime_error("Future layers not evicted farthest first");
                previous_future = event.layer;
            }
        }
        if (!seen_future) throw std::runtime_error("Pressure fixture did not exercise future eviction");
        memory.release(workspace);
        const int next = model.group(3, tidal::WeightPart::Attention);
        memory.prefetch({next});
        if (!stats.prefill_weight_prefetches) throw std::runtime_error("Missing asynchronous prefetch");
        if (model.resident(model.group(0, tidal::WeightPart::Attention)))
            throw std::runtime_error("Completed layer reloaded before prefill ended");
        memory.protect(3, {next});
        bool rejected = false;
        try { memory.acquire(resident); }
        catch (const std::runtime_error & error) {
            rejected = std::string(error.what()).find("Insufficient VRAM for active prefill") != std::string::npos;
        }
        if (!rejected || memory.workspace_bytes()) throw std::runtime_error("Impossible workspace request was not rejected cleanly");
        memory.restore();
        for (const auto & [name, tensor] : model.weights) {
            std::vector<char> bytes(ggml_nbytes(tensor)); ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
            if (bytes != originals.at(name)) throw std::runtime_error("Restoration after OOM changed weights");
        }
        std::cout << "Residency priority, release/reload bytes, pending-copy lifetime and aliases passed\n";
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
