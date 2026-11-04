# tidal.cpp

A C++20 / ggml inference prototype for a **tiny, random-weight Dense derivative
of Qwen3.8-Flash-Next (`qwen4exp`)**. It implements stateful autoregressive decoding
with GPU Gated DeltaNet and QSA indexing, and CPU sparse attention over a BF16 KV cache.

This is an initial correctness milestone, not a trained language model or a loader
for the released MoE checkpoint. Token IDs are synthetic; generated IDs have no
linguistic meaning. PLE, MoE routing, vision, MTP and batched prefill are outside this
version. Prompt ingestion uses the same one-token state transitions as decoding.

| Work / storage | `--backend cuda` | `--backend cpu` |
|---|---|---|
| Quantized or unquantized matrix weights | GPU | CPU |
| GDN convolution, gates, recurrent state and output | GPU | CPU |
| Gated Residual, Q/K/V projections, RoPE, Dense FFN, output head | GPU | CPU |
| QSA raw-key pooling, compressed BF16 index cache, scores and top-k | GPU | CPU |
| Main BF16 KV cache, selected QK, stable softmax, weighted V | **CPU** | CPU |

CUDA is the default execution request. Missing CUDA or an unsupported operation
causes an error; there is no automatic CPU fallback. The CPU mode is an explicit
reference/testing mode. Only selected block IDs and the current Q/K/V cross to the
host; the attention output returns to the GPU. The full main KV cache stays in RAM.

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

## Generate and run

The generator writes a complete GGUF directly, with the same seeded source tensors
for every weight type. Embeddings stay BF16; normalization, convolution and decay
parameters stay F32. Matrix formats: **F32, F16, BF16, Q8_0 and Q4_0**.

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

The CLI reports the backend, CPU attention calls, synchronized graph/transfer times
and bytes. JSON contains every step's logits and selected token IDs. `--repeat 2`
resets and replays the same sequence to check deterministic state reset. Timing is
diagnostic; graph allocation and thread creation are intentionally unoptimized.

## Verify

```sh
python -m pip install numpy
# Reconfigure if NumPy was installed after the first configure.
cmake -S . -B build
ctest --test-dir build --output-on-failure

# Run the same numerical and placement checks on a real CUDA device.
python tests/test_reference.py --bin-dir build-cuda/bin --backend cuda
```

Tests compare complete multi-token logits and QSA selections against an independent
NumPy GGUF reader/dequantizer and model implementation. They cover all five formats,
complete/incomplete blocks, generation, reset, two superblocks, stable sparse GQA,
invalid input, truncated files, and strict backend rejection. Low-precision tests
allow documented activation-rounding error; F32 has a much tighter tolerance.
The oracle models CPU and CUDA operand conversions separately, checked against
isolated native matvecs. Generated tokens, reset/replay QSA selections and exact
transfer-byte budgets are also verified. Test coverage and per-format error
bounds are documented in [the validation contract](docs/validation.md).

CI builds and tests the CPU implementation and compiles/links the CUDA path for
SM 7.5. The CUDA CI job's numerical tests use **CPU mode** because hosted runners
have no GPU. Real GPU runtime correctness and performance require the command above.

See [architecture and scope](docs/architecture.md) and [validation record](docs/validation.md).
The project remains Apache-2.0; ggml is a separate MIT-licensed dependency.
