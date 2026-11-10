#include "tidal/tidal.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool value, const char * error) { if (!value) throw std::runtime_error(error); }
void close(const tidal::Step & expected, const tidal::Step & actual) {
    require(expected.position==actual.position && expected.logits.size()==actual.logits.size(), "GPU KV positions/logit count differ");
    double error=0, norm=0;
    for (size_t i=0;i<expected.logits.size();++i) {
        require(std::isfinite(actual.logits[i]) && std::isfinite(expected.logits[i]), "Non-finite GPU KV logits");
        error+=std::pow(double(expected.logits[i])-actual.logits[i],2);
        norm+=double(expected.logits[i])*expected.logits[i];
    }
    require(std::sqrt(error/std::max(norm,1e-20))<.004, "GPU KV prefill/decode/extension relative error");
    require(expected.selections.size()==actual.selections.size(), "GPU KV selection count differs");
    for (size_t i=0;i<expected.selections.size();++i)
        require(expected.selections[i].layer==actual.selections[i].layer && expected.selections[i].tokens==actual.selections[i].tokens,
                "GPU QSA lost causal/block selection semantics");
}
}

int main(int argc, char ** argv) {
    try {
        if (argc!=3) throw std::runtime_error("Usage: tidal-gpu-kv-test MODEL fp16|bf16");
        tidal::Options options{"cuda",0,2};
        options.kv_placement=tidal::KVPlacement::GPU;
        options.prefill_precision=std::string(argv[2])=="bf16" ? tidal::PrefillPrecision::BF16 : tidal::PrefillPrecision::FP16;
        options.prefill_chunk_tokens=5;options.prefill_attention_tile=3;
        options.profile=tidal::ProfileMode::Nodes;
        tidal::Engine engine(argv[1],options);
        const auto & c=engine.config();
        int layers=0;for(int l=0;l<c.layers;++l) layers+=!c.is_recurrent(l);
        const int dim=c.head_dim<=64 ? 64 : c.head_dim<=128 ? 128 : 256;
        const uint64_t bytes=uint64_t(layers)*2*dim*((c.context+255)/256*256)*c.kv_heads*2;
        require(engine.statistics().gpu_kv_bytes==bytes, "Incorrect persistent GPU KV byte accounting");
        std::vector<int32_t> tokens;
        for(int i=0;i<33;++i) tokens.push_back((i*17+1)%c.vocab);
        const auto expected=engine.prefill(tokens);
        const auto continuation=engine.decode(17);
        bool flash=false;
        for(const auto & e:engine.profile_events()) {
            if(e.category=="node" && e.op=="FLASH_ATTN_EXT") {
                require(e.backend.rfind("CUDA",0)==0, "GPU KV attention ran on CPU");
                require(e.input_types.size()>=3 && e.input_types[1]==(options.prefill_precision==tidal::PrefillPrecision::FP16 ? "f16" : "bf16"),
                        "GPU KV FlashAttention uses the wrong storage precision");
                flash=true;
            }
            require(e.name!="D2H_async" && e.name!="kv_append_bf16", "GPU KV exported/appended to CPU");
        }
        require(flash && engine.statistics().gpu_attention_calls==uint64_t(layers), "GPU decode attention was not recorded");
        engine.set_profile(tidal::ProfileMode::Off);
        engine.reset();
        for(size_t i=0;i<tokens.size();++i) close(expected[i],engine.decode(tokens[i]));
        close(continuation,engine.decode(17));
        engine.reset();
        size_t offset=0;
        for(int n:{1,3,2,6,21}) {
            for(const auto & step:engine.prefill({tokens.begin()+offset,tokens.begin()+offset+n})) close(expected[offset++],step);
        }
        close(continuation,engine.decode(17));
        engine.reset();
        for(size_t i=0;i<7;++i) close(expected[i],engine.decode(tokens[i]));
        const auto suffix=engine.prefill({tokens.begin()+7,tokens.end()});
        for(size_t i=0;i<suffix.size();++i) close(expected[7+i],suffix[i]);
        close(continuation,engine.decode(17));
        engine.reset();
        const auto compact=engine.prefill(tokens,false);
        for(const auto & step:compact) require(step.logits.empty() && step.selections.empty(), "Compact GPU KV prefill retained diagnostics");
        const auto step=engine.decode(17,false);
        require(step.logits.empty() && step.selections.empty(), "Compact GPU KV decode downloaded selection/logits");
        const auto & s=engine.statistics();
        require(!s.full_attention_calls && !s.sparse_attention_calls && !s.prefill_kv_to_host_bytes, "GPU KV mode used CPU attention/cache");
        require(s.device_to_host_bytes==8, "GPU KV compact path downloaded more than two finiteness scalars");
        require(s.gpu_kv_bytes==bytes, "Reset lost GPU KV accounting");
        {
            // Compact decode runs a different (single-graph) path than the
            // diagnostic one; both must pick the same tokens, across a 256 boundary.
            auto compact_options=options;compact_options.profile=tidal::ProfileMode::Off;
            tidal::Engine diagnostic(argv[1],compact_options), fast(argv[1],compact_options);
            diagnostic.prefill(tokens);fast.prefill(tokens,false);
            const int steps=std::min<int>(300,c.context-int(tokens.size()));
            for(int i=0;i<steps;++i) {
                const auto expected_token=diagnostic.sample(), actual_token=fast.sample();
                require(expected_token==actual_token, "Compact GPU KV decode changed greedy sampling");
                diagnostic.decode(expected_token);
                require(fast.decode(actual_token,false).logits.empty(), "Compact GPU KV decode downloaded logits");
            }
            fast.reset();diagnostic.reset();
            for(size_t i=0;i<tokens.size();++i) {fast.decode(tokens[i],false);diagnostic.decode(tokens[i]);}
            require(fast.sample()==diagnostic.sample(), "Compact GPU KV decode after reset changed sampling");
        }
        if(c.context>=514) {
            options.prefill_chunk_tokens=37;options.prefill_attention_tile=17;options.profile=tidal::ProfileMode::Off;
            tidal::Engine whole(argv[1],options), extended(argv[1],options);
            std::vector<int32_t> prompt;for(int i=0;i<513;++i) prompt.push_back(tokens[i%tokens.size()]);
            const auto target=whole.prefill(prompt);
            auto prefix=extended.prefill({prompt.begin(),prompt.begin()+255});
            close(target[255],extended.decode(prompt[255]));
            auto tail=extended.prefill({prompt.begin()+256,prompt.end()});
            for(size_t i=0;i<prefix.size();++i) close(target[i],prefix[i]);
            for(size_t i=0;i<tail.size();++i) close(target[256+i],tail[i]);
            close(whole.decode(17),extended.decode(17));
            extended.reset();
            auto replay=extended.prefill(prompt);
            for(size_t i=0;i<replay.size();++i) close(target[i],replay[i]);
        }
        if(c.context>=4352 && c.architecture==tidal::Architecture::Qwen4Exp &&
           options.prefill_precision==tidal::PrefillPrecision::BF16) {
            // Cross the backend's sparse-gather length threshold with wide GQA.
            // Hybrid prefill uses dense masked FlashAttention; decode uses an
            // independent CPU selected-only dot/softmax/V implementation.
            auto baseline_options=options;baseline_options.profile=tidal::ProfileMode::Off;
            baseline_options.prefill_chunk_tokens=0;baseline_options.prefill_attention_tile=0;
            tidal::Engine sparse(argv[1],baseline_options);
            baseline_options.kv_placement=tidal::KVPlacement::CPU;
            tidal::Engine dense(argv[1],baseline_options);
            std::vector<int32_t> prompt;for(int i=0;i<4093;++i) prompt.push_back(tokens[i%tokens.size()]);
            sparse.prefill(prompt,false);dense.prefill(prompt,false);
            require(sparse.sample()==dense.sample(), "Sparse GPU KV prefill changed sampling");
            for(int i=0;i<21;++i) close(dense.decode(tokens[i]),sparse.decode(tokens[i]));
        }
        options.prefill_memory_limit=1;
        tidal::Engine impossible(argv[1],options);
        bool rejected=false,poisoned=false;
        try { impossible.prefill(tokens); } catch(const std::runtime_error &) { rejected=true; }
        try { impossible.decode(1); } catch(const std::runtime_error &) { poisoned=true; }
        require(rejected && poisoned, "GPU KV failure silently fell back or retained valid partial state");
        impossible.reset();engine.reset();close(engine.decode(1),impossible.decode(1));
        std::cout << "GPU KV precision, prefill/decode/prefix/reset, causal selection, transfers and failure recovery passed\n";
    } catch(const std::exception & e) {std::cerr << e.what() << '\n';return 1;}
}
