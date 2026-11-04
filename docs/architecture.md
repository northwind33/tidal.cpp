# Tiny hybrid decoder

## Reference and deliberate reductions

The user's Qwen3-Flash-Next refers here to Qwen3.8-Flash-Next, exposed as
`qwen4_exp` in Transformers and `qwen4exp` in llama.cpp. The released model is MoE;
this fixture follows the earlier Dense-engine plan. It retains the decoder's
GDN/QSA rhythm, Gated Residual reads/writes, Q/K normalization, partial NeoX RoPE,
gated attention output and SwiGLU. MoE is replaced by a Dense SwiGLU, and PLE,
vision and MTP are disabled. There is no tokenizer or claim of checkpoint compatibility.

Sources inspected on 2026-11-04:

- [Qwen's model repository](https://github.com/QwenLM/Qwen3.8-Flash-Next).
- [Transformers model equations](https://github.com/huggingface/transformers/blob/f2754311602414ed561cdf8ed9d852390f514404/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py),
  source blob `6b9402efb45edff7854e54efc715a4a50e2383fa`.
- [llama.cpp graph implementation](https://github.com/ggml-org/llama.cpp/blob/4b29d02629b76160350ded907c64b733d98a54d1/src/models/qwen4exp.cpp),
  source blob `e4848cb77f26c52e97acb42e1240da377e0404c3`.
- [Pinned ggml](https://github.com/ggml-org/ggml/tree/facdc44a67da17df4433adcc468b424f48e9ded2),
  used directly for backend allocation, GGUF, quantization and graph kernels.

The project implements the model equations independently. It does not vendor the
Transformers or llama.cpp source files. The separate NumPy oracle is test code;
the engine and dummy-model generator are C++.

## One decoding step

1. Gather the BF16 embedding on the selected backend and duplicate it across the
   residual streams for this token.
2. At each GDN layer, normalize each residual stream, compute low-rank read gates,
   average the gated reads, and project Q/K/V, output gate, decay and delta gate.
   Apply the depthwise causal convolution with previous-token history. L2-normalize
   Q and K, repeat-interleave key heads to value heads, then use ggml's fused GDN
   recurrence. Save the F32 recurrence and convolution history on the same backend.
3. RMS-normalize/gate the GDN result, project it out, inject it into the residual
   streams, then perform a separately gated Dense FFN read and write.
4. At each QSA layer, project/normalize/rotate Q/K and project V. Keep query gates
   and residual injection gates on the backend. The QSA indexer projects raw keys,
   averages each complete micro-block, applies key RMSNorm and RoPE at the block's
   start position, then stores the compressed key in a BF16 device cache.
5. Widen that compressed cache to F32 inside the backend graph. Score each block
   by the sum over index heads of `relu(q · k) / sqrt(index_dim)`. Run top-k on
   the backend. Expand the selected block IDs on the host and always append the
   incomplete tail. Block scores and full caches never transfer to the CPU.
6. Download current Q/K/V and block IDs. Append K/V to the CPU BF16 cache and compute
   selected-only GQA QK, stable softmax and weighted V on CPU threads. Each query
   head uses the corresponding contiguous group of KV heads.
7. Upload the attention output, apply the attention output gate/projection, residual
   write and FFN on the backend. A final learned residual read and output matrix
   produce the logits.

Completed blocks are ranked normally; only an *incomplete* tail is unconditionally
included. A token can attend to itself when selected, and no operation can read a
future token. Selection lists are sorted and checked for duplicates before attention.
Equal top-k scores can select different tied blocks across backends; fixtures avoid
ties at the budget boundary. There is no random sampling: generation uses argmax.

## Precision and storage

| Tensor / operation | Precision |
|---|---|
| Embeddings | BF16 |
| Matrix weights | F32, F16, BF16, Q8_0 or Q4_0 |
| Norm gamma, convolution weights, A-log and dt bias | F32 |
| Graph activations, GDN recurrent/conv state, raw index ring | F32 |
| Compressed index cache, host main KV cache | BF16 |
| Sparse attention scores, softmax, weighted values | F32 |

GGUF norm tensors contain the effective gamma, i.e. the reference's `(1 + weight)`
is already folded, except GDN's gated RMSNorm which directly uses its gamma. Qwen
L2 normalization adds epsilon *inside* the square root; the implementation uses
scaled RMSNorm instead of ggml's different clamp-based L2 operation.

Q/K/V tensors use F32 transfers in this prototype, then K/V are rounded to BF16 on
the host. Model weights and GDN state never round-trip through host memory during
decoding. Quantized matrices remain quantized in backend memory; ggml executes
their native quantized matvecs. The BF16 index cache is the only stored matrix
widened for scoring. BF16 is a storage format here, not a requirement for native
BF16 GPU arithmetic.

## GGUF contract and API

The architecture tag is `tidal-qwen4exp-dense`, schema `tidal.version = 1`.
All configuration keys use `tidal.*`. Tensor names and shapes are centralized in
`src/model.cpp`; ggml dimensions are `[input, output]`, reversed relative to the
usual row-major matrix shape. Unknown architectures, metadata versions, missing
tensors, shape/type mismatches and truncated payloads are rejected.

One Engine owns one sequence. `decode` advances it once, `prefill` repeats those
transitions, and `reset` clears all GDN/index state, logical CPU KV lengths and
statistics. Exceeding the configured capacity raises an error; no sliding-window
eviction occurs. Failed graph execution poisons the sequence until `reset`, so a
partially updated state cannot silently be reused. Independent Engines may run
independently; an individual Engine is not thread-safe.

## Current performance limits

Graphs are allocated per phase, transfers synchronize, CPU head workers are created
per attention call, and top-k materializes a score for every compressed block. There
is no graph reuse, transfer overlap, batched prefill, server, multi-sequence cache,
checkpoint converter or long-context benchmark claim. These are explicit next
optimization/integration boundaries after correctness and GPU runtime validation.
