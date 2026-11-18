// Optional independent integration oracle. Build against llama.cpp, not tidal.
#include "llama.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    try {
        if (argc != 5) throw std::runtime_error("Usage: qwen35-llama-reference MODEL tokenize|trace|detokenize INPUT OUTPUT");
        const std::string mode = argv[2];
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        mp.vocab_only = mode != "trace";
        auto * model = llama_model_load_from_file(argv[1], mp);
        if (!model) throw std::runtime_error("Reference model load failed");
        const auto * vocab = llama_model_get_vocab(model);
        std::ofstream output(argv[4], std::ios::binary);
        if (!output) throw std::runtime_error("Cannot open reference output");
        if (mode == "tokenize") {
            const std::string text = argv[3];
            std::vector<llama_token> tokens(text.size() + 32);
            const int n = llama_tokenize(vocab, text.data(), int(text.size()), tokens.data(), int(tokens.size()), false, true);
            if (n < 0) throw std::runtime_error("Tokenization buffer too small");
            for (int i = 0; i < n; ++i) output << (i ? "," : "") << tokens[i];
        } else {
            std::vector<llama_token> tokens;
            std::istringstream input(argv[3]);
            for (std::string token; std::getline(input, token, ',');) tokens.push_back(std::stoi(token));
            if (mode == "detokenize") {
                std::vector<char> piece(4096);
                for (auto token : tokens) {
                    const int n = llama_token_to_piece(vocab, token, piece.data(), int(piece.size()), 0, true);
                    if (n < 0) throw std::runtime_error("Token piece too large");
                    output.write(piece.data(), n);
                }
            } else if (mode == "trace") {
                auto cp = llama_context_default_params();
                cp.n_ctx = uint32_t(std::max<size_t>(256, tokens.size()));
                cp.n_batch = cp.n_ubatch = cp.n_seq_max = 1;
                cp.n_threads = cp.n_threads_batch = 4;
                cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
                cp.offload_kqv = false;
                cp.type_k = cp.type_v = GGML_TYPE_BF16;
                auto * ctx = llama_init_from_model(model, cp);
                if (!ctx) throw std::runtime_error("Reference context creation failed");
                auto batch = llama_batch_init(1, 0, 1);
                batch.n_tokens = 1; batch.n_seq_id[0] = 1; batch.seq_id[0][0] = 0; batch.logits[0] = 1;
                for (size_t i = 0; i < tokens.size(); ++i) {
                    batch.token[0] = tokens[i]; batch.pos[0] = llama_pos(i);
                    if (llama_decode(ctx, batch)) throw std::runtime_error("Reference decode failed");
                    output.write(reinterpret_cast<const char *>(llama_get_logits_ith(ctx, 0)),
                                 std::streamsize(llama_vocab_n_tokens(vocab)) * sizeof(float));
                }
                llama_batch_free(batch);
                llama_free(ctx);
            } else throw std::runtime_error("Unknown reference mode");
        }
        output.close();
        if (!output) throw std::runtime_error("Reference write failed");
        llama_model_free(model);
        llama_backend_free();
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
