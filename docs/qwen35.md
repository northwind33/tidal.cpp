# Qwen3.5-0.8B text inference

The engine loads the public model's native `qwen35` GGUF directly. Standard full
attention QK, softmax and weighted V execute on CPU over a BF16 KV cache in CPU
RAM during decoding. CUDA prefill uses [layer-major GPU attention](prefill.md),
with temporary GPU KV and asynchronous export to CPU RAM. Embedding lookup, all projections, normalization, RoPE, output gates,
residual additions, Dense FFNs, Gated DeltaNet and the output head execute on the
requested CUDA backend. Unsupported CUDA operations fail; there is no scheduler
that moves neural graph operations to CPU. Tokenization and orchestration are host work; token sampling defaults to the
model backend.

This supports single-sequence text generation. Image/video inputs and vision
projectors, MTP execution, scaled RoPE and `qwen35moe` are not
implemented. Other Qwen3.5 sizes are not covered by the public-model validation.
The existing synthetic `qwen4exp` Dense/MoE/PLE paths remain available.

## Download and run

Build the CUDA preset as described in the README. Download into the ignored
`models/` directory; these examples pin public source revisions:

```sh
mkdir -p models/qwen35
qwen35_gguf_revision=807dfd9a6fadbf703207aac3bfb0b2b3040d077c
qwen35_tokenizer_revision=2fc06364715b967f1860aea9cf38778875588b17
for format in BF16 Q8_0; do
  curl -fL --retry 3 \
    "https://huggingface.co/ggml-org/Qwen3.5-0.8B-GGUF/resolve/$qwen35_gguf_revision/Qwen3.5-0.8B-$format.gguf" \
    -o "models/qwen35/Qwen3.5-0.8B-$format.gguf.partial" &&
  mv "models/qwen35/Qwen3.5-0.8B-$format.gguf.partial" "models/qwen35/Qwen3.5-0.8B-$format.gguf"
done
curl -fL --retry 3 \
  "https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/$qwen35_tokenizer_revision/tokenizer.json" \
  -o models/qwen35/tokenizer.json

# Optional text-only frontend dependency; inference still uses tidal-cli.
python3 -m venv build/text-venv
build/text-venv/bin/python -m pip install tokenizers
build/text-venv/bin/python tools/qwen35_text.py \
  --model models/qwen35/Qwen3.5-0.8B-Q8_0.gguf \
  --tokenizer models/qwen35/tokenizer.json \
  --context 4096 --steps 64 \
  --prompt 'What is 2 + 2? Reply with just the number.'
```

Only one GGUF is needed for generation; both are used by the integration test.
The frontend applies single-turn ChatML with thinking disabled by default, using
the official tokenizer. `--thinking` allows a thinking section; `--raw` accepts
literal completion text. It does not run a second inference engine or download
anything. It removes its temporary full-logit trace after decoding the output.

For applications that already have token IDs, use `tidal-cli --tokens ...` or
`Engine::prefill` / `Engine::decode`. `--context N` limits allocated sequence
capacity to N, up to the GGUF maximum. The public model declares a large default;
choose the context actually needed. Prompt length plus `--steps` must fit.
The CLI stops after a generated GGUF EOS token by default; `--stop-at-eos 0`
requests a fixed number of steps. Prompt EOS tokens do not end prompt ingestion.
The Engine API itself always processes exactly the supplied tokens.

`tidal-bench` accepts the same GGUF and `--context`, including node profiles and
NVTX ranges. Its fixed-length workload intentionally continues through EOS.
Profiles distinguish `full.prepare`, CPU `full.attention` QK/softmax/V and
`full.finish`, alongside GPU GDN, embedding and head phases.

## Native layout and execution

The public text decoder has 24 main layers: 18 GDN layers and six full-attention
layers, with hidden width 1024, FFN width 3584 and vocabulary 248320. Full attention
uses eight Q heads, two KV heads, width 256 and 64 rotary dimensions. GDN has 16
key/value heads of width 128 and a four-tap convolution.

The loader reads `qwen35.*` metadata, including the full-attention interval or
explicit recurrent-layer mask. Tensor shapes/types, required metadata and all
payload bounds are checked before backend weight allocation. The implementation
handles these differences from the synthetic schema:

- Native tensor names map to the existing internal projection names.
- GGUF norm gains already contain the effective `(1 + weight)` where applicable.
  `ssm_a` already contains `-exp(A_log)`; it must not be exponentiated again.
- Native GDN value heads use tiled broadcast order. Synthetic repeated-head
  ordering is preserved separately.
- Full attention uses every position from zero through the current token, with
  ordinary residual additions and SwiGLU; no sparse indexer or Gated Residual.
- Equal text positions on the three interleaved MRoPE axes reduce to partial
  NeoX RoPE. This does not implement multimodal position handling.
- Missing `output.weight` means a tied output head sharing `token_embd.weight`.
  Both GPU lookup and matvec support are checked, with one resident allocation.
- `block_count` includes the declared `nextn_predict_layers` tail. Recognized
  MTP tensors are excluded from normal autoregressive execution and reported as
  `skipped_tensors` / `skipped_weight_bytes`; arbitrary extra tensors are rejected.

The existing per-tensor storage policy also applies to native matrices. Supported
formats remain encoded on the backend; unsupported types are explicitly widened
at load time. Public integration coverage is BF16 and Q8_0. Norms, convolution
weights and decay parameters use F32 at runtime. Activations and GDN state are
F32. During decoding, K/V round to BF16 on CPU, and queries, scores, softmax and
value accumulation remain F32. Current Q/K/V download and attention results
upload. During prefill, the active layer’s KV is temporarily on GPU; GPU attention
uses the selected working format and exports canonical BF16 KV for decoding.

## Verification

Routine tests need no public weights or tokenizer:

```sh
ctest --preset dev-cuda
python3 tests/test_qwen35.py --bin-dir build/dev-cuda/bin --backend cuda
```

The fixture suite covers native metadata, independent NumPy logits, all causal
prefixes, tied and untied heads, unequal GDN head counts, reset/chunked state,
MTP exclusion, EOS, invalid files and node/transfer placement. CTest runs its CPU
mode; the explicit command validates real CUDA execution.

The public-model suite uses the downloaded files and writes local artifacts only:

```sh
python3 tests/test_qwen35_real.py \
  --bin-dir build/dev-cuda/bin --model-dir models/qwen35
```

It checks both formats over an arithmetic chat prompt plus greedy generation and
EOS, exact reset replay, full-prefix selections, transfer bytes and actual-model
node placement. Every first-token matrix product is also checked against NumPy
with identical F32 input vectors, using atol/rtol 2e-5. Full BF16 trajectories
require per-token vocabulary NRMSE below 0.5%. Q8_0 has a coarser 10% trajectory
guard, combined with the tight isolated matrix checks and identical NumPy argmax
at every tested token. Tiny reductions can cross Q8 activation-code or FP16-scale
boundaries and amplify through the trained model. This Q8 guard does not establish
BF16-level logit agreement or a universal quantization error bound.

For a separate implementation check, build the optional helper against an external
llama.cpp checkout, not this project's ggml:

```sh
git clone https://github.com/ggml-org/llama.cpp build/qwen35-validation/llama.cpp
git -C build/qwen35-validation/llama.cpp checkout 8bed7cfb999cebc03d2cc5b78911026a8d57549a
cmake -S build/qwen35-validation/llama.cpp -B build/qwen35-validation/llama-build \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF
cmake --build build/qwen35-validation/llama-build --target llama-server -j 4
c++ -std=c++17 -O2 tests/qwen35_llama_reference.cpp \
  -Ibuild/qwen35-validation/llama.cpp/include \
  -Ibuild/qwen35-validation/llama.cpp/ggml/include \
  -Lbuild/qwen35-validation/llama-build/bin \
  -Wl,-rpath,"$PWD/build/qwen35-validation/llama-build/bin" \
  -lllama -lggml -lggml-base -o build/qwen35-validation/llama-reference
python3 tests/test_qwen35_real.py \
  --bin-dir build/dev-cuda/bin --model-dir models/qwen35 \
  --llama-reference build/qwen35-validation/llama-reference
```

This independently checks the generated answer using CPU llama.cpp with BF16 KV.
It does not require identical full logits: llama.cpp's CPU attention matmuls also
round queries/probabilities, unlike tidal's F32 CPU attention. The separate NumPy
decoder models tidal's storage and arithmetic contract. Short tests do not establish
long-context quality or multimodal compatibility. Model downloads, reference
checkouts, timing reports and traces stay under ignored `models/` and `build/`;
local hardware inventories and measurement results do not belong in commits.

## Sources

- [Official model configuration](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/config.json).
- [Public GGUF repository](https://huggingface.co/ggml-org/Qwen3.5-0.8B-GGUF/tree/807dfd9a6fadbf703207aac3bfb0b2b3040d077c).
- [Transformers equations](https://github.com/huggingface/transformers/blob/f2754311602414ed561cdf8ed9d852390f514404/src/transformers/models/qwen3_5/modeling_qwen3_5.py).
- [GGUF conversion](https://github.com/ggml-org/llama.cpp/blob/4b29d02629b76160350ded907c64b733d98a54d1/conversion/qwen.py).
