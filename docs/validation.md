# Validation record

Initial local validation, 2026-11-04: Linux x86-64, GCC 13.3, CMake 4.4.4,
ggml `facdc44a67da17df4433adcc468b424f48e9ded2`, CPU AVX2 build with `GGML_NATIVE=OFF`.

## Numerical results

Each format uses seed 42, hidden 128, three GDN layers plus one QSA layer,
24 fixed prompt tokens and six greedily generated tokens. Every step's entire
256-logit vector and selected QSA token IDs are compared to the separate NumPy
implementation. The same sequence is reset/replayed and checked for exact logits.

| Matrix format | Maximum absolute logit difference | QSA selected IDs |
|---|---:|---|
| F32 | 0.00010186425 | Match at every position |
| F16 | 0.0018075559 | Match at every position |
| BF16 | 0.0096533003 | Match at every position |
| Q8_0 | 0.020599455 | Match at every position |
| Q4_0 | 0.020034760 | Match at every position |

These compare each file with its own dequantized weights, **not** Q4/Q8 logits
against the original F32 model. ggml's CPU low-precision matvecs also round their
activation operands. The oracle independently emulates this conversion; small
reduction differences near rounding thresholds can create a discrete change that
propagates through recurrent states. Tests set separate per-format absolute and
relative bounds (F32: 2e-4; F16: 3e-3; BF16: 0.012; Q4/Q8: 0.025), and require
normalized vocabulary RMS error below 2.5%. These are numerical implementation
checks on random fixtures, not model quality or accuracy benchmarks.

The suite also passes with eight layers, hidden 256, seed 123 and two independent
QSA caches. Core tests verify incomplete-tail selection, block expansion, duplicate
and future-index rejection, stable softmax at large logits, GQA head mapping,
selected-only V aggregation, and stale-KV exclusion after reset. CLI tests check
invalid tokens/options, context limits, malformed/truncated GGUF and missing CUDA.

## Reproduce

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
python tests/test_reference.py --bin-dir build/bin
```

Both CTest suites pass locally. An additional Debug build enables AddressSanitizer
and UndefinedBehaviorSanitizer for the tidal library and its executables. The
managed executor blocks LeakSanitizer's process inspection, so leak detection must
be disabled here; it is not evidence that the leak check passed.

## CUDA boundary

The implementation selects exactly one ggml backend for all model/indexer graphs
and checks every node with `ggml_backend_supports_op`. CPU sparse attention is a
separate C++ implementation with CPU-owned BF16 storage. CUDA requests never fall
back to CPU. The test runner can assert the reported CUDA backend and nonzero
host/device transfers.

No CUDA device or toolkit is available in the initial editing environment. The
GitHub `cuda-compile` job builds/links the SM 7.5 path but runs numerical tests in
CPU mode. **A passing CUDA build does not establish GPU runtime correctness or
performance.** The remaining hardware acceptance command is:

```sh
python tests/test_reference.py --bin-dir build-cuda/bin --backend cuda
```

No tokens/second, long-context speedup or Titan RTX runtime result is claimed.

The first CUDA CI run compiled ggml's CUDA kernels and the engine, then failed
linking the executables: the GPU-less container had no `libcuda.so.1` for ggml's
VMM driver-API symbols. The workflow now sets `GGML_CUDA_NO_VMM=ON` for that CI job.
The GPU development container retains the normal VMM default; NVIDIA Container
Toolkit supplies the host driver libraries at runtime. See
[the Dev Container guide](devcontainer.md) for setup and hardware validation.
