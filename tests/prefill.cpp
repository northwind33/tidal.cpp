#include "tidal/tidal.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <stdexcept>

namespace {
void require(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }
void close(const tidal::Step & a, const tidal::Step & b, float tolerance) {
    require(a.position == b.position && a.logits.size() == b.logits.size(), "Prefill position/logit count differs");
    double square = 0, norm = 0;
    for (size_t i=0; i<a.logits.size(); ++i) {
        require(std::isfinite(a.logits[i]) && std::isfinite(b.logits[i]), "Non-finite prefill logits");
        square += std::pow(double(a.logits[i])-b.logits[i],2); norm += double(a.logits[i])*a.logits[i];
    }
    if (std::sqrt(square/std::max(norm,1e-20)) > tolerance)
        throw std::runtime_error("Prefill/chunk/decode relative error exceeds " + std::to_string(tolerance) + " at " + std::to_string(a.position));
    require(a.selections.size()==b.selections.size(), "Attention selection layer count differs");
    for (size_t i=0; i<a.selections.size(); ++i)
        require(a.selections[i].layer==b.selections[i].layer && a.selections[i].tokens==b.selections[i].tokens,
                "Causal attention selection differs");
}
}
int main(int argc, char ** argv) {
    try {
        if (argc != 3) throw std::runtime_error("Usage: tidal-prefill-test f32-model.gguf cpu|cuda");
        tidal::Options options{argv[2],0,2}; options.prefill_attention_tile=7; options.profile=tidal::ProfileMode::Nodes;
        tidal::Engine engine(argv[1],options);
        const std::vector<int32_t> tokens{1,7,23,9,42,5,6,2,77,13,84,11,90,3,42,28,61,7,0,93,34,22,1,255};
        auto expected=engine.prefill(tokens);
        const auto stats=engine.statistics();
        const bool gpu=options.backend=="cuda";
        if (gpu) {
            int attention_layers=0;
            for (int l=0;l<engine.config().layers;++l) attention_layers += !engine.config().is_recurrent(l);
            require(stats.prefill_layers==uint64_t(engine.config().layers) && stats.prefill_tokens==tokens.size(), "Missing batch layers/tokens");
            require(!stats.full_attention_calls && !stats.sparse_attention_calls, "Prefill ran CPU attention");
            require(stats.prefill_gpu_attention_calls==uint64_t(attention_layers*((tokens.size()+6)/7)), "GPU attention tiles missing");
            require(stats.prefill_kv_to_host_bytes==uint64_t(attention_layers)*tokens.size()*engine.config().kv_heads*engine.config().head_dim*4,
                    "Asynchronous KV payload differs");
            int current=-1,flash=0; bool batched_mm=false,transfer=false;
            for (const auto & e:engine.profile_events()) {
                if (e.category=="graph" && e.layer>=0 && e.layer<engine.config().layers && e.phase.rfind("prefill.",0)==0) {
                    require(e.layer>=current,"Prefill traversed backwards to an earlier layer");current=e.layer;
                }
                if (e.category=="node") {
                    require(e.backend.rfind("CUDA",0)==0,"Prefill node ran outside CUDA");
                    flash += e.op=="FLASH_ATTN_EXT";
                    if (e.op=="MUL_MAT" && e.shape.size()>1 && e.shape[1]==int64_t(tokens.size())) batched_mm=true;
                }
                transfer |= e.name=="D2H_async" && e.phase=="prefill.kv";
            }
            require(flash==int(stats.prefill_gpu_attention_calls) && batched_mm && transfer,"Batch kernels or async transfer not profiled");
        }
        const auto continuation=engine.decode(17);
        engine.reset();
        for (size_t i=0;i<tokens.size();++i) close(expected[i],engine.decode(tokens[i]),gpu?.004f:0.f);
        close(continuation,engine.decode(17),gpu?.004f:0.f);
        engine.reset();
        size_t offset=0;
        for (int n:{1,3,2,6,12}) {
            auto chunk=engine.prefill({tokens.begin()+offset,tokens.begin()+offset+n});
            for (const auto & step:chunk) close(expected[offset++],step,gpu?.004f:0.f);
        }
        close(continuation,engine.decode(17),gpu?.004f:0.f);
        engine.reset();
        const auto replay=engine.prefill(tokens);
        for (size_t i=0;i<tokens.size();++i) close(expected[i],replay[i],0);
        const auto sampled=engine.sample();
        engine.reset();
        const auto compact=engine.prefill(tokens,false);
        require(engine.sample()==sampled,"Compact prefill changed token selection");
        for (const auto & step:compact) require(step.logits.empty() && (!gpu || step.selections.empty()),"Compact prefill downloaded diagnostics");
        close(continuation,engine.decode(17),gpu?.004f:0.f);
        require(engine.prefill({}).empty(),"Empty prefill changed sequence");
        if (gpu) {
            // Internal sublayer ranges must preserve layer order, recurrence,
            // PLE n-grams/convolution, causal QSA and CPU-KV decode continuation.
            for (int width : {1,5,11}) {
                auto split_options=options;
                split_options.prefill_chunk_tokens=width;
                split_options.prefill_attention_tile=0;
                tidal::Engine split(argv[1],split_options);
                auto actual=split.prefill(tokens);
                for (size_t i=0;i<tokens.size();++i) close(expected[i],actual[i],.004f);
                const auto s=split.statistics();
                require(s.prefill_layers==uint64_t(split.config().layers) && s.prefill_tokens==tokens.size(),"Internal ranges changed layer/token counts");
                require(s.prefill_chunk_max_tokens<=uint64_t(width) && s.prefill_chunk_min_tokens>=1,"Explicit range bound ignored");
                require(!s.full_attention_calls && !s.sparse_attention_calls,"Internal ranges used CPU attention");
                int current=-1;
                std::map<std::string,int> positions;
                for (const auto & e:split.profile_events()) {
                    if (e.category=="graph" && e.phase.rfind("prefill.",0)==0 && e.layer>=0) {
                        require(e.layer>=current,"Internal ranges crossed a layer boundary");current=e.layer;
                    }
                    if (e.category=="prefill" && e.name=="chunk") {
                        auto & next=positions[e.phase+":"+std::to_string(e.layer)];
                        require(e.shape.size()==1 && e.position==next,"Chunk profile lost its absolute token range");
                        next+=int(e.shape[0]);
                    }
                }
                for (const auto & [phase,next]:positions) {
                    (void)phase;
                    require(next==int(tokens.size()),"Chunk profile omitted or duplicated prompt tokens");
                }
                close(continuation,split.decode(17),.004f);
            }
            options.profile=tidal::ProfileMode::Phases;
            options.prefill_memory_limit=engine.model_info().weight_buffer_bytes;
            tidal::Engine pressured(argv[1],options);
            const auto actual=pressured.prefill(tokens);
            // Pressure can change the automatic GEMM partition and reduction order.
            for (size_t i=0;i<tokens.size();++i) close(expected[i],actual[i],.004f);
            const auto s=pressured.statistics();
            require(s.prefill_weight_evictions && s.prefill_weight_reloads,"Pressure did not evict/reload weights");
            bool restored=false;
            for (const auto & e:pressured.profile_events()) {
                if (e.phase=="prefill.restore") restored=true;
                if (restored) require(e.category!="graph","Weights restored before prefill graph completion");
            }
            close(continuation,pressured.decode(17),.004f);
            options.prefill_memory_limit=1;
            tidal::Engine tiny(argv[1],options);
            bool rejected=false;
            try { tiny.prefill(tokens); }
            catch (const std::runtime_error & e) { rejected=std::string(e.what()).find("Insufficient VRAM")!=std::string::npos; }
            require(rejected,"Impossible GPU budget did not fail clearly");
            bool poisoned=false;
            try { tiny.decode(1); } catch(const std::runtime_error &) { poisoned=true; }
            require(poisoned,"Failed prefill did not poison partial state");
            tiny.reset(); engine.reset(); close(engine.decode(1),tiny.decode(1),0);
        }
        if (gpu && engine.config().context>=514) {
            auto extended_options=options;
            extended_options.prefill_memory_limit=0;
            extended_options.prefill_chunk_tokens=37;
            extended_options.prefill_attention_tile=17;
            extended_options.profile=tidal::ProfileMode::Off;
            tidal::Engine whole(argv[1],extended_options), extended(argv[1],extended_options);
            std::vector<int32_t> long_tokens;
            for (int i=0;i<513;++i) long_tokens.push_back(tokens[i%tokens.size()]);
            auto target=whole.prefill(long_tokens);
            auto prefix=extended.prefill({long_tokens.begin(),long_tokens.begin()+300});
            auto suffix=extended.prefill({long_tokens.begin()+300,long_tokens.end()});
            for (size_t i=0;i<prefix.size();++i) close(target[i],prefix[i],.004f);
            for (size_t i=0;i<suffix.size();++i) close(target[300+i],suffix[i],.004f);
            close(whole.decode(17),extended.decode(17),.004f);
        }
        std::cout << "Layer order, GPU attention, KV export, prefix continuation, reset, pressure and failure recovery passed\n";
    } catch(const std::exception & e) {std::cerr<<e.what()<<'\n';return 1;}
}
