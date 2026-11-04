# Validation contract

The numerical suites compare the tiny Dense decoder against an independent
NumPy GGUF reader and model implementation. Machine-specific measurements,
hardware/software inventories and local test logs are not stored in this file.

## Reproduce

```sh
cmake --build --preset dev-cuda
ctest --preset dev-cuda
python3 tests/test_reference.py --bin-dir build/dev-cuda/bin --backend cuda
```

CTest's numerical reference suite uses CPU execution, including in a CUDA build.
The explicit Python command exercises CUDA model/indexer graphs with CPU sparse
attention. A CUDA compile alone does not establish GPU runtime correctness.

A CPU-only build can use:

```sh
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF
cmake --build build-cpu -j
ctest --test-dir build-cpu --output-on-failure
```

## Numerical coverage

Each format uses seed 42, hidden 128, three GDN layers plus one QSA layer,
24 fixed prompt tokens and six greedily generated tokens. Every step's entire
256-logit vector and selected QSA token IDs are compared to the NumPy reference
with the requested backend's arithmetic. Generated tokens must equal the oracle's
argmax. Reset/replay requires identical positions, logits and QSA selections.

Tests compare each GGUF to its own weights and native operand conversions, not
quantized logits to the original F32 model. Low-precision CPU and CUDA execution
can produce different trajectories. This validates implementation on synthetic
fixtures, not trained-model quality or universal error bounds.

Absolute and relative tolerances used by `assert_allclose` are:

| Matrix format | CPU atol / rtol | CUDA atol / rtol |
|---|---:|---:|
| F32 | 2e-4 | 4e-4 |
| F16 | 3e-3 | 1e-3 |
| BF16 | 0.012 | 4e-4 |
| Q8_0 / Q4_0 | 0.025 | 0.025 |

Every token additionally requires normalized vocabulary RMS error below 2.5%.
F32 reductions can cross BF16 cache rounding boundaries even with F32 matrix
weights. The eight-layer, hidden-256, seed-123 fixture exercises two independent
QSA caches and reset/replay, using atol/rtol 3e-4.

## Backend-specific reference arithmetic

- CUDA F16/BF16 matvecs with F32 accumulation consume F32 activations. CPU dot
  products convert those activations to the corresponding floating-point type.
- Q4/Q8 dot products accumulate integer products within blocks before applying
  scales. Dequantizing both operands before a floating-point matrix product
  changes reduction rounding and can alter subsequent activation quantization.
- CUDA Q8_1 uses half-away-from-zero rounding and stores the original activation
  sum in FP16. Q4_0's unsigned-nibble correction uses that sum. CPU Q8_0 uses no
  such sum; the AVX2 path rounds to nearest-even.

`tidal-matvec-test` runs native ggml matvecs on GGUF weights and F32 vectors supplied
by Python. It checks five representative matrices per format with zero blocks,
mixed values and positive/negative halfway and near-halfway inputs, at atol/rtol
2e-5. The full model oracle remains pure NumPy and does not call this executable
to calculate expected logits.

Cases execute independently, so a failed format does not suppress other formats,
multi-layer checks or malformed-input tests. To select one format:

```sh
python3 tests/test_reference.py --bin-dir build/dev-cuda/bin --backend cuda --formats q4_0
```

Core tests cover incomplete tails, block expansion, stable softmax, grouped KV
heads, selected-only V aggregation and stale-KV exclusion after reset. CLI tests
cover invalid inputs, capacity limits, malformed/truncated GGUF and strict CUDA
rejection. Dev Container regression tests stub GPU preflight commands; they do
not establish GPU correctness.

## Placement and CI

All model/indexer graphs use exactly one backend and check every node with
`ggml_backend_supports_op`; CUDA requests have no automatic CPU fallback. CPU
sparse attention owns the main BF16 KV cache. Tests check backend identity,
attention calls and the exact transfer-byte budget derived from model dimensions.
Only current Q/K/V, selected block IDs and logits download; token/RoPE positions
and attention outputs upload. Full main KV caches remain in CPU RAM.

The GitHub CUDA job builds SM 7.5 and runs numerical tests in CPU mode on GPU-less
runners. It uses `GGML_CUDA_NO_VMM=ON` to avoid a host-driver linking dependency.
The GPU development container uses normal VMM and host drivers supplied by
NVIDIA Container Toolkit. Real GPU validation is a separate explicit test run.
