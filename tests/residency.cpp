#include "residency.hpp"
#include "prefill_memory.hpp"
#include "prefill_tensor.hpp"
#include "prefill_chunks.hpp"
#include "kv_export.hpp"
#include "ggml-cpu.h"
#ifdef TIDAL_CUDA
#include "ggml-cuda.h"
#endif
#include <iostream>
#include <set>
#include <cstring>

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
        // A multi-group prefetch must preserve requested groups already resident.
        // Constrain the budget to exactly the two requested future groups.
        const int future_ffn = model.group(3, tidal::WeightPart::FeedForward);
        options.prefill_memory_limit = model.groups[next].allocation_bytes + model.groups[future_ffn].allocation_bytes;
        tidal::PrefillMemory paired(model, backend.get(), stats, profile, options);
        model.evict(next);
        paired.prefetch({next, future_ffn});
        model.wait_all();
        if (!model.resident(next) || !model.resident(future_ffn) || model.resident_bytes() > options.prefill_memory_limit)
            throw std::runtime_error("Prefetch evicted a requested group or exceeded the reserved budget");
        paired.restore();
        // Decode restoration must not keep a prefill-only workspace reserve:
        // all temporary buffers and the prefill kernel pool have been released.
        auto restore_options=options;
        restore_options.prefill_memory_reserve=UINT64_MAX-1;
        tidal::PrefillMemory restoration(model,backend.get(),stats,profile,restore_options);
        model.evict(last);
        restoration.restore();
        if (!model.resident(last) || model.resident_bytes()!=resident)
            throw std::runtime_error("Prefill-only headroom prevented decode weight restoration");
        // Exercise reduction with real backend allocations under a managed cap,
        // not only simulated exceptions. Two rows fit; eight and four must fail.
        auto adaptive_options=options;
        adaptive_options.prefill_memory_limit=model.groups[active].allocation_bytes+8192;
        adaptive_options.prefill_memory_reserve=0;
        tidal::PrefillMemory adaptive(model,backend.get(),stats,profile,adaptive_options);
        adaptive.protect(2,{active});
        std::vector<int> committed, reduced;
        tidal::visit_prefill_chunks(13,8,true,[&](int,int n) {
            tidal::PrefillTensorStorage storage(adaptive,backend.get());
            storage.tensor(GGML_TYPE_F32,1024,n);
            storage.allocate();
            if (!model.resident(active)) throw std::runtime_error("Adaptive allocation evicted active weights");
        },[&](int first,int n) {
            for (int i=first;i<first+n;++i) committed.push_back(i);
        },[&](int,int,int smaller) { reduced.push_back(smaller); });
        if (reduced!=std::vector<int>{4,2} || committed.size()!=13 || adaptive.workspace_bytes())
            throw std::runtime_error("Backend allocation pressure did not shrink and release ranges");
        for (int i=0;i<13;++i)
            if (committed[i]!=i) throw std::runtime_error("Backend allocation retry duplicated tokens");
        adaptive.restore();
        adaptive_options.prefill_memory_limit=model.groups[active].allocation_bytes;
        tidal::PrefillMemory impossible(model,backend.get(),stats,profile,adaptive_options);
        impossible.protect(2,{active});
        int attempts=0; bool minimum_rejected=false;
        try {
            tidal::visit_prefill_chunks(8,8,true,[&](int,int n) {
                ++attempts;
                tidal::PrefillTensorStorage storage(impossible,backend.get());
                storage.tensor(GGML_TYPE_F32,1024,n); storage.allocate();
            },[](int,int) { throw std::runtime_error("Impossible range was committed"); },[](int,int,int) {});
        } catch (const tidal::PrefillAllocationError &) { minimum_rejected=true; }
        if (!minimum_rejected || attempts!=4 || impossible.workspace_bytes())
            throw std::runtime_error("Backend allocation did not reject after shrinking to one token");
        impossible.restore();
        if (std::string(argv[2]) == "cuda") {
            options.prefill_memory_limit = 0;
            tidal::PrefillMemory export_memory(model, backend.get(), stats, profile, options);
            auto source = std::make_shared<tidal::PrefillTensorStorage>(export_memory, backend.get());
            // More than one 32-MiB K/V staging chunk, with an existing prefix.
            constexpr int width=1024, count=17000, prefix=3;
            auto * keys=source->tensor(GGML_TYPE_BF16,width,count);
            auto * values=source->tensor(GGML_TYPE_BF16,width,count);
            source->allocate();
            std::vector<ggml_bf16_t> original_keys(size_t(width)*count), original_values(size_t(width)*count);
            for (size_t i=0;i<original_keys.size();++i) {
                original_keys[i].bits=uint16_t(0x3f00+i%128);
                original_values[i].bits=uint16_t(0xbf80+i%128);
            }
            ggml_backend_tensor_set(keys,original_keys.data(),0,ggml_nbytes(keys));
            ggml_backend_tensor_set(values,original_values.data(),0,ggml_nbytes(values));
            tidal::SparseAttention destination(1,1,width,count,1);
            destination.append_raw(reinterpret_cast<const uint16_t*>(original_keys.data()),
                                   reinterpret_cast<const uint16_t*>(original_values.data()),prefix);
            const auto before=stats.prefill_kv_to_host_bytes;
            std::weak_ptr<tidal::PrefillTensorStorage> lifetime=source;
            tidal::KVExporter exporter(0,stats,profile);
            exporter.enqueue(source,keys,values,destination,prefix,count-prefix,3);
            source.reset();
            if (lifetime.expired() || !export_memory.workspace_bytes())
                throw std::runtime_error("KV transfer lost its GPU source ownership");
            exporter.finish();
            if (!lifetime.expired() || export_memory.workspace_bytes() || destination.size()!=count ||
                stats.prefill_kv_to_host_bytes-before!=uint64_t(count-prefix)*width*4 ||
                [&] {
                    std::vector<ggml_bf16_t> k(original_keys.size()), v(original_values.size());
                    destination.copy_rows(0,count,reinterpret_cast<uint16_t*>(k.data()),reinterpret_cast<uint16_t*>(v.data()));
                    return std::memcmp(k.data(),original_keys.data(),k.size()*sizeof(ggml_bf16_t)) ||
                           std::memcmp(v.data(),original_values.data(),v.size()*sizeof(ggml_bf16_t));
                }())
                throw std::runtime_error("Chunked asynchronous KV export changed bytes, prefix or lifetime");
            // FP16 CPU KV: a device-side overflow (Inf) must fail the export, not be stored.
            auto f16 = std::make_shared<tidal::PrefillTensorStorage>(export_memory, backend.get());
            auto * fk=f16->tensor(GGML_TYPE_F16,width,4), * fv=f16->tensor(GGML_TYPE_F16,width,4);
            f16->allocate();
            std::vector<uint16_t> half_keys(size_t(width)*4, 0x3c00), half_values(half_keys.size(), 0xbc00);
            half_values[size_t(width)*2+5]=0x7c00;
            ggml_backend_tensor_set(fk,half_keys.data(),0,ggml_nbytes(fk));
            ggml_backend_tensor_set(fv,half_values.data(),0,ggml_nbytes(fv));
            tidal::SparseAttention half(1,1,width,4,1,nullptr,GGML_TYPE_F16);
            bool rejected=false;
            exporter.enqueue(f16,fk,fv,half,0,4,5);
            try { exporter.finish(); } catch (const std::overflow_error & e) {
                rejected=std::string(e.what()).find("layer 5")!=std::string::npos && half.size()==0;
            }
            if (!rejected) throw std::runtime_error("FP16 KV export accepted an overflowed row");
        }
        std::cout << "Residency priority, release/reload bytes, pending-copy lifetime and aliases passed\n";
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
