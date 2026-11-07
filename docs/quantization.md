# Quantization and mixed GGUF storage

Support follows the pinned ggml revision in CMakeLists.txt. `tidal-dummy --list-types 1`
enumerates the 28 weight storage types: F32, F16, BF16 and 25 quantized formats.

- Q1_0, Q2_0, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0.
- Q2_K, Q3_K, Q4_K, Q5_K, Q6_K.
- IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS.
- TQ1_0, TQ2_0, MXFP4, NVFP4.

Q8_1/Q8_K are ggml activation/dot-product work formats, without GGUF weight
encoders/decoders in this revision. They are not accepted as model weight formats.
Removed repacked layouts and integer tensor types are also rejected. A format's
block size must divide the tensor's first dimension, including each expert slice.

Each GGUF tensor carries its own type; no single file-wide quantization format is
assumed. Loader support applies to matrices, expert stacks and embeddings. Norm,
convolution and scalar parameters accept the same storage types when aligned,
and are converted to F32 for their elementwise operations. Generator defaults
keep these small tensors F32 and token/PLE tables BF16.

The loader probes the operation required by each tensor on the selected backend.
Supported matrix/lookup types remain encoded in backend memory. Unsupported types
are decoded to F32 once at load time and uploaded to that backend. For example,
pinned CUDA lacks TQ matrix kernels and NVFP4 row-gather kernels. This conversion
increases resident memory; it does not execute inference graphs on CPU or change
the selected backend. Conversion messages identify every affected tensor.

CLI and benchmark reports include parameters, stored/resident weight bytes,
allocated weight-buffer bytes and converted tensor count. These counters exclude
graph scratch, recurrent state, CUDA workspaces and driver allocations; actual
peak VRAM must also be checked for large-model fit.

## Generating mixed weights

`--type` selects the default matrix format. `--tensor-types PATH` reads ordered
`glob=type` rules. Globs match entire tensor names, support `*` and `?`, and later
matches override earlier ones. A comment starts with `#`. Every rule must match
at least one tensor. Type names are case-insensitive.

```sh
build/dev-cuda/bin/tidal-dummy --preset quant-test --type q6_K \
  --tensor-types rules.txt --output models/mixed.gguf
```

The writer uses row-aligned chunks, validates encoded data, pads GGUF offsets and
publishes the completed file from an exclusive `.partial` file without replacing
an existing output. Caught failures remove that partial file; abrupt process
termination can leave it for inspection. Source RNG order is independent of chunk
size and quantization thread count. Loading also uses bounded row chunks, avoiding
a whole-tensor or whole-file staging copy.

The dummy generator supplies unit importance weights when ggml requires an
imatrix. This is explicit synthetic calibration, not representative activation
statistics from a trained model. Per-tensor storage rules offer the mechanism used
by mixed-precision GGUFs, not Unsloth Dynamic's automatic sensitivity decisions.

## Verification

```sh
python3 tests/test_quant.py --bin-dir build/dev-cuda/bin --backend cuda
```

The suite places every type in Dense matrices and MoE expert stacks, checks exact
uploaded bytes (or exact load-time dequantization), and compares native matvecs to
decoded-weight dot products. Quantized matvec checks bound native activation
quantization error; they do not compare quantized weights to original F32 quality.
Mixed models execute finite-logit generation and reset/chunked state checks.
Tests also cover conversion for non-native lookup formats, mixed norm storage,
chunk/thread determinism, malformed rules, alignment and overwrite rejection.
`quantization` runs the CPU suite in CTest; CUDA runtime validation is explicit.
