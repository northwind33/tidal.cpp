# tidal.cpp

A C++20 / ggml inference prototype supporting **public Qwen3.5-0.8B GGUF text
inference** and **random-weight Dense/MoE derivatives of Qwen3.8-Flash-Next
(`qwen4exp`)**. Gated DeltaNet runs on GPU. By default, ordinary full attention for Qwen3.5,
or QSA-selected sparse attention for the synthetic model, reads a BF16 KV cache
in CPU RAM and computes QK/softmax/V on CPU during decoding. CUDA prefill instead
processes the whole prompt layer by layer, with GPU attention and asynchronous
KV export to CPU RAM; see [prefill](docs/prefill.md).
Within each layer, automatic memory-aware token ranges bound temporary tensors;
`--prefill-chunk N` and `--prefill-tile N` select explicit sizes for comparisons.
For benchmarks, `--kv-placement gpu` retains all KV on CUDA and runs attention
there during both prefill and decoding. `--prefill-precision fp16|bf16` selects
the persistent GPU KV format too. Choose a `--context` that fits device memory;
see the [GPU KV mode](docs/prefill.md#persistent-gpu-kv-benchmark-mode).

Qwen3.5 loads native `qwen35` GGUFs without conversion to the synthetic schema.
See [the real-model guide](docs/qwen35.md) for downloads, text generation and
independent verification. The `tidal-dummy` models still have random weights and
synthetic token IDs. Their optional PLE implements n-gram embedding, gating and
dilated convolution; optional MoE implements top-k routing, sparse SwiGLU experts
and a gated shared expert. Released `qwen4exp` checkpoints, vision and MTP execution
remain outside this version. CUDA prompt ingestion uses a dedicated layer-major
batch path; the CPU backend retains sequential reference execution.

The following placement table describes **default decoding**. CUDA prefill keeps the
active layer’s KV and attention computation on GPU temporarily.

| Work / storage | `--backend cuda` | `--backend cpu` |
|---|---|---|
| Quantized or unquantized matrix weights | GPU | CPU |
| GDN convolution, gates, recurrent state and output | GPU | CPU |
| PLE embedding table, lookup, projections, gates, dilated convolution and state | GPU | CPU |
| PLE n-gram hashing and recent token IDs | CPU | CPU |
| Gated Residual, Q/K/V projections, RoPE, Dense FFN, output head | GPU | CPU |
| MoE routing, selected experts and gated shared expert | GPU | CPU |
| QSA raw-key pooling, compressed BF16 index cache, scores and top-k | GPU | CPU |
| Main BF16 KV cache, full-prefix or selected QK, stable softmax, weighted V | **CPU** | CPU |

CUDA is the default execution request. Missing CUDA or an unsupported operation
causes an error; there is no automatic CPU fallback. The CPU mode is an explicit
reference/testing mode. In the default CPU KV mode, current Q/K/V cross to the host; QSA additionally downloads
selected block IDs. The attention output returns to the GPU. The full main KV cache
stays in RAM. Tokenization and host orchestration use CPU. Token selection defaults
to the model backend; an explicit CPU sampler remains available for comparison.

## Build

For development on the Titan RTX server, a CUDA-enabled VS Code Dev Container is
included. Connect to the server using Remote - SSH, open this repository there,
then run **Dev Containers: Reopen in Container**. The container provides CUDA
12.8.1, CMake, Ninja, GDB, ccache and NumPy, and configures the `dev-cuda` preset
for SM 7.5. See [the setup and debugging guide](docs/devcontainer.md) for host
prerequisites, build commands and real-GPU checks.

Requirements: CMake 3.24+, a C++20 compiler, Git, and optionally the CUDA toolkit.
ggml is fetched at a pinned commit (see `CMakeLists.txt`). For an existing checkout,
pass `-DTIDAL_GGML_SOURCE_DIR=/path/to/ggml` at that same revision.

```sh
# CPU reference build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# CUDA build, including Titan RTX (SM 7.5)
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DTIDAL_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build-cuda -j
```

For another NVIDIA GPU, use its compute capability instead of `75`. The graph uses
F32 activations/recurrent state and ggml conversions for BF16 storage, so it does
not require native BF16 tensor cores.

## Generate and run synthetic models

The generator streams a complete GGUF directly, with the same seeded source tensors
for every weight type. Embeddings default to BF16; normalization, convolution and
decay parameters default to F32. All **25 ggml weight quantization formats** plus
F32/F16/BF16 are accepted, including K-quants, IQ, TQ, MXFP4/NVFP4 and Q1_0/Q2_0.
Use `tidal-dummy --list-types 1` for the pinned version's exact list and block sizes.
`--preset quant-test` provides dimensions aligned for all these formats.

```sh
mkdir -p models
build/bin/tidal-dummy --output models/tiny-f32.gguf --type f32
build/bin/tidal-dummy --output models/tiny-q4.gguf --type q4_0

# Explicit CPU reference execution
build/bin/tidal-cli --model models/tiny-f32.gguf --backend cpu \
  --tokens 1,7,23,9,42,5,6,2,77,13,84,11,90 --steps 8 --json build/trace.json

# Requested CPU/GPU split
build-cuda/bin/tidal-cli --model models/tiny-q4.gguf --backend cuda \
  --tokens 1,7,23,9,42,5,6,2,77,13,84,11,90 --steps 8 --json build-cuda/trace.json
```

The default is 1,063,672 parameters: hidden 128, FFN 256, four residual streams,
three GDN layers and one QSA layer, vocabulary/context 256. QSA uses four-token
blocks and a two-block budget, so even a short test exercises actual sparsity.
`--hidden 256 --layers 8 --context 4096` creates a larger test fixture.
`--preset dense-27b` constructs a 27.108B Dense model without PLE. A mixed 6.128-bpw
recipe and an opt-in 24-GiB CUDA/CPU integration test are described in
[the large-model guide](docs/large-model.md).

To include PLE in the GGUF (layer IDs are **zero-based**, GDN layers only):

```sh
build/dev-cuda/bin/tidal-dummy --output models/tiny-ple.gguf --type q4_0 --ple-layers 0,2
build/dev-cuda/bin/tidal-cli --model models/tiny-ple.gguf --backend cuda \
  --tokens 0,1,7,23,0,9,42,5,6,2,77,13 --steps 8 --repeat 2
```

PLE tables default to BF16; `--ple-type` selects any supported matrix format.
`--ple-ngram`, `--ple-heads`, `--ple-dim`, `--ple-conv`, `--ple-eos` and
`--ple-vocab-base` control synthetic PLE dimensions. Small prime vocabulary ranges
keep fixtures manageable. Older PLE-free GGUFs retain their behavior. This extends
the `tidal-qwen4exp-dense` contract. `--experts 4 --experts-used 2` instead writes
`tidal-qwen4exp-moe`; `--expert-ffn`, `--shared-ffn` (0 disables the shared expert)
and `--norm-topk 0|1` configure the experts. Released `qwen4exp` GGUFs still require
a converter into the engine's tensor and metadata contract.

The CLI reports the backend, CPU attention calls, synchronized graph/transfer times
and bytes. JSON contains every step's logits and selected token IDs by default;
`--logits 0` keeps only token IDs and selections. `--repeat 2`
resets and replays the same sequence to check deterministic state reset. Timing is
diagnostic; graph allocation and thread creation are intentionally unoptimized.

For per-tensor mixed quantization, pass `--tensor-types rules.txt`. Each line is
`tensor_glob=type`, with `*`/`?` wildcards, comments beginning with `#`, and the last
matching rule taking precedence. Unmatched rules and incompatible block widths
are errors. For example:

```text
*.ffn.*=q5_K
*.attn.*=q6_K
*.attn.*_norm=f32
token_embd.weight=q8_0
output.weight=q6_K
```

This provides mixed storage within one GGUF; it does not reproduce Unsloth's
calibration or sensitivity-based format selection. IQ formats requiring an imatrix
receive uniform importance weights in the **synthetic dummy generator**.
`--dry-run 1` reports parameter count, encoded bytes and effective bpw without
writing; `--chunk-mib 16 --threads 4` controls streaming generation.
See [format support and conversion](docs/quantization.md) for backend behavior.

## Benchmark and profile

`tidal-bench` reports prefill/decode/total tokens per second with warmup and repeated
measurements. Detailed profiling runs separately so it does not contaminate those
throughput samples:

```sh
build/dev-cuda/bin/tidal-bench --model models/tiny-q4.gguf --backend cuda \
  --prompt 24 --steps 32 --warmup 2 --repetitions 10 \
  --profile nodes --json build/dev-cuda/bench.json --trace build/dev-cuda/trace.json
```

Profiles include per-layer/ggml-node synchronized wall times, tensor shapes/types,
transfers, graph allocation, and CPU QK/softmax/V stages. NVTX ranges connect to
Nsight; `tools/compare_bench.py` compares saved reports and can flag regressions.
See [sampling](docs/sampling.md) for temperature, top-k, top-p, min-p, penalties,
seeding and `--sampler cpu` comparisons.

See [benchmarking and profiling](docs/benchmark.md) for measurement semantics,
baseline comparisons, and the distinction between wall timings and CUDA kernel metrics.

## Serve

`tidal-server` keeps one model loaded behind an HTTP API that OpenAI clients
(`/v1/chat/completions`, `/v1/completions`), Anthropic clients (`/v1/messages`) and
llama.cpp server clients (`/completion`, `/tokenize`, `/props`, `/slots`, `/metrics`, ...)
can all use, with streaming, thinking, tool calls and prompt-cache reuse between
requests. It reads the tokenizer from the GGUF, so real Qwen GGUFs need no Python
frontend. A test page is served at `/`.

```sh
build-cuda/bin/tidal-server --model models/qwen35/Qwen3.5-0.8B-Q8_0.gguf --context 8192
curl http://127.0.0.1:8080/v1/chat/completions \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 64}'
```

See [the server guide](docs/server.md) for options, endpoints and limits.

## Verify

```sh
python -m pip install numpy
# Reconfigure if NumPy was installed after the first configure.
cmake -S . -B build
ctest --test-dir build --output-on-failure

# Run the same numerical and placement checks on a real CUDA device.
python tests/test_reference.py --bin-dir build-cuda/bin --backend cuda
python tests/test_qwen35.py --bin-dir build-cuda/bin --backend cuda
python tests/test_prefill.py --bin-dir build-cuda/bin --backend cuda
```

Tests compare complete multi-token logits and QSA selections against an independent
NumPy GGUF reader/dequantizer and model implementation. The original numerical suite covers five representative formats,
complete/incomplete blocks, generation, reset, two superblocks, stable sparse GQA,
invalid input, truncated files, and strict backend rejection. Low-precision tests
allow documented activation-rounding error; F32 has a much tighter tolerance.
The oracle models CPU and CUDA operand conversions separately, checked against
isolated native matvecs. Generated tokens, reset/replay QSA selections and exact
transfer-byte budgets are also verified. Test coverage and per-format error
bounds are documented in [the validation contract](docs/validation.md).
Native Qwen3.5 has separate fixture and opt-in public-model tests, including full
causal attention, tied embeddings, MTP exclusion and CUDA/CPU placement checks.

CI builds and tests the CPU implementation and compiles/links the CUDA path for
SM 7.5. The CUDA CI job's numerical tests use **CPU mode** because hosted runners
have no GPU. Real GPU runtime correctness and performance require the command above.

See [architecture and scope](docs/architecture.md) and [validation record](docs/validation.md).
The project remains Apache-2.0; ggml is a separate MIT-licensed dependency.
