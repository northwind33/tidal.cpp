# Hybrid decoder

## Architectures

The engine dispatches on the GGUF architecture. Native `qwen35` supports the
public Qwen3.5-0.8B text decoder; `tidal-qwen4exp-dense` and `tidal-qwen4exp-moe`
retain the synthetic model contract below. Both default to GPU model graphs with CPU
BF16 KV storage and CPU QK/softmax/V during decoding. CUDA prefill uses a
[layer-major GPU path](prefill.md) with temporary device KV, asynchronous host
export and weight eviction/reload. The explicit CPU backend remains available
for reference tests.

The optional `--kv-placement gpu` benchmark mode keeps a persistent, padded
FP16/BF16 KV allocation per attention layer on CUDA, using the selected
`--prefill-precision` for both phases. It allocates no CPU main KV cache and
exports no KV after prefill. Full attention and QSA use the same ggml
FlashAttention primitive as prefill; QSA pooling, ranking and masking stay on
device. A finite-entry bound enables ggml's sparse KV gather when the backend
and shape support it. Diagnostic selection IDs may be read back on request;
compact inference transfers only the existing finiteness/sampling scalars.

For Qwen3.5, the ordinary attention layers attend to every cached position through
the current token, without QSA indexing or top-k. Q/K/V projections, per-head
normalization, partial text RoPE, sigmoid output gates, residual additions, Dense
FFNs and the tied output head run on GPU. GDN convolution and recurrent state also
stay on GPU. There is one residual stream, and no Gated Residual mixing or PLE.
Native GGUF stores negative exponentiated GDN decay and tiled value-head ordering;
the synthetic schema's A-log and repeated head layout remain unchanged.

Declared MTP tail tensors are validated for payload bounds and skipped during
standard autoregressive inference. The tied embedding/output tensor is uploaded
once. [The Qwen3.5 guide](qwen35.md) specifies native metadata, source references,
text generation and validation limits.

## Synthetic qwen4exp reference and deliberate reductions

The user's Qwen3-Flash-Next refers here to Qwen3.8-Flash-Next, exposed as
`qwen4_exp` in Transformers and `qwen4exp` in llama.cpp. The released model is MoE;
this fixture follows the earlier Dense-engine plan. It retains the decoder's
GDN/QSA rhythm, Gated Residual reads/writes, Q/K normalization, partial NeoX RoPE,
gated attention output and SwiGLU. The FFN can be Dense or MoE. PLE is optional;
vision and MTP are disabled. These synthetic models have no tokenizer or claim of
released `qwen4exp` checkpoint compatibility.

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

## One synthetic qwen4exp decoding step (default CPU KV)

1. Gather the BF16 embedding on the selected backend and duplicate it across the
   residual streams for this token.
2. Apply PLE, when configured, before the layer's attention residual read (see below).
   At each GDN layer, normalize each residual stream, compute low-rank read gates,
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
ties at the budget boundary. Generation defaults to greedy device sampling, with
temperature/top-k/top-p/min-p and history penalties available. The sampler is
separate from QSA block selection; see [sampling](sampling.md).

## Optional per-layer embedding (PLE)

The equations follow `Qwen4ExpTextNGramEmbedding` and `Qwen4ExpTextPLELayer` in the
pinned Transformers source above. Each configured GDN layer owns its own table,
hash constants, projections, norm weights and convolution history:

1. For n-gram orders 2 through N, XOR the signed-int64-safe products of token IDs
   and per-layer multipliers. Each head takes the remainder by its vocabulary size
   and adds its table offset. Missing history is EOS; a preceding EOS masks itself
   and earlier tokens. The current EOS still uses its preceding context.
2. Gather and concatenate one embedding row per head. Project a key for each
   residual branch and a shared value. RMS-normalize the key and current residual
   separately within each branch of width `hidden`.
3. Compute `s = dot(key, query) / sqrt(hidden)` and gate the shared value by
   `sigmoid(sign(s) * sqrt(max(abs(s), 1e-6)))` in each branch.
4. Group-normalize the gated value, then apply a causal depthwise convolution with
   dilation N and SiLU. Add both the gated value and convolution output to the
   residual before the usual GDN/FFN operations.

The host keeps at most N-1 token IDs and computes the integer hashes. Row indices
upload once per PLE layer/token; lookup and all floating-point operations execute
on the selected backend. The table is not copied back to the host. The F32
convolution history holds `(kernel-1)*N` vectors per layer on the backend. EOS
resets the hash context only, not convolution, GDN or attention state. `reset()`
clears all of them. Kernel size 1 has no convolution history.

Profiling exposes a `ple` phase/layer, host `ngram_hash`, embedding `lookup`, key/value
projections, grouped norms, gate, individual convolution taps and SiLU nodes.

## Mixture of experts

With `tidal-qwen4exp-moe`, every FFN uses the reference's softmax router, top-k
experts, optional normalization of selected probabilities and sigmoid-gated
shared SwiGLU expert. Routing, gathering probabilities and `MUL_MAT_ID` expert
operations run on the selected backend. Expert IDs are unique and remain on the
device. Only selected experts execute; no host routing readback is needed.

`tidal.experts`, `experts_used`, `expert_ffn`, `shared_ffn` are UINT32 metadata;
`tidal.norm_topk` is BOOL. All are required for the MoE tag and rejected on a Dense
model. Expert counts are bounded to 256, with 1..experts selected per token.
For each `blk.L.ffn.` prefix, `router` is [hidden, experts], `gate_exps` and
`up_exps` are [hidden, expert_ffn, experts], and `down_exps` is
[expert_ffn, hidden, experts]. With shared_ffn > 0, `shared.gate`, `shared.up`,
`shared.down` are ordinary Dense matrices and `shared_gate` is [hidden, 1].
Router weights default to F32. Expert operations use the backend's native
accumulation policy; pinned CUDA rejects forced-F32 accumulation on MUL_MAT_ID.

## Precision and storage

This table describes default decoding. [GPU prefill](prefill.md) adds temporary device KV
and FP16/BF16 working storage for FlashAttention.

| Tensor / operation | Precision |
|---|---|
| Token/PLE embeddings | Any supported weight storage format (BF16 default) |
| Matrix weights, including expert stacks | Any supported weight storage format |
| Norm gamma, convolution weights, A-log and dt bias | F32 |
| Graph activations, GDN recurrent/conv state, PLE conv state, raw index ring | F32 |
| Compressed index cache | BF16 |
| Host main KV cache | BF16 (default) or FP16 with `cpu_kv_precision` |
| Full or sparse attention scores, softmax, weighted values | F32 |

GGUF norm tensors contain the effective gamma, i.e. the reference's `(1 + weight)`
is already folded, except GDN's gated RMSNorm which directly uses its gamma. Qwen
L2 normalization adds epsilon *inside* the square root; the implementation uses
scaled RMSNorm instead of ggml's different clamp-based L2 operation.

Q/K/V tensors use F32 transfers in this prototype, then K/V are rounded to BF16 on
the host. Model weights and GDN state never round-trip through host memory during
decoding. Supported quantized matrices remain encoded in backend memory; ggml
executes their native matvecs. Unsupported backend types are explicitly converted
to F32 at load time, as described in [quantization support](quantization.md).
The BF16 index cache is widened for scoring. BF16 is a storage format here, not a requirement for native
BF16 GPU arithmetic.

F32 accumulation does not make operand conversion identical across backends.
CPU F16/BF16 dot products round activations to those types, whereas the CUDA
single-vector paths retain F32 activations. CPU Q4/Q8 matvecs quantize activations
to Q8_0; CUDA uses Q8_1, including an FP16 original-activation sum used in Q4_0's
zero-point correction. The NumPy oracle models these paths separately, including
integer block dot products. Low-precision CPU/CUDA logits and greedy sequences
can therefore differ; see [the validation contract](validation.md).

## GGUF contract and API

Synthetic architecture tags are `tidal-qwen4exp-dense` and `tidal-qwen4exp-moe`, with schema `tidal.version = 1`.
All configuration keys use `tidal.*`. Tensor names and shapes are centralized in
`src/model.cpp`; ggml dimensions are `[input, output]`, reversed relative to the
usual row-major matrix shape. Unknown architectures, metadata versions, missing
tensors, shape/type mismatches and truncated payloads are rejected.

Dimension validation supports the synthetic 27B preset as well as tiny fixtures;
tensor element/byte products use 64-bit sizes. Loader payload bounds are checked
before backend allocation, and file data is streamed in bounded row chunks.
The [large-model guide](large-model.md) specifies the hypothetical Dense layout,
mixed-format recipe and opt-in runtime checks.

PLE is an optional extension of version 1: old files omit the entire `tidal.ple.*`
group. A present group requires these keys; partial/unknown PLE keys are rejected:

| Metadata | GGUF type / meaning |
|---|---|
| `tidal.ple.layers` | UINT32 array of sorted, unique, zero-based GDN layer IDs |
| `tidal.ple.ngram_size`, `heads_per_ngram`, `embedding_dim`, `conv_kernel`, `eos_token_id` | UINT32 scalars (each key uses the `tidal.ple.` prefix) |
| `tidal.ple.L.vocab` | UINT32 padded table row count for decoder layer L |
| `tidal.ple.L.multipliers` | UINT64 array, N entries |
| `tidal.ple.L.offsets`, `tidal.ple.L.vocab_sizes` | UINT64 arrays, H = (N-1)*heads_per_ngram entries |

Multipliers must keep every valid token product within signed int64. Head ranges
must be contiguous, nonempty and within the table; remaining rows are padding.
Runtime uses stored constants, allowing externally initialized hash tables.
The dummy writer uses the original SplitMix64 derivation with `--seed`, PLE module
ordinal (not decoder layer ID), successive prime head sizes and 128-row padding.
Transformers' `ple_layer_ids` are one-based and must be converted when authoring
this schema. Tensor layout is:

| Tensor prefix `blk.L.ple.` | ggml dimensions |
|---|---|
| `embedding` | [embedding_dim / H, vocab] |
| `key`, `value` | [embedding_dim, hidden * branches], [embedding_dim, hidden] |
| `norm_key`, `norm_query`, `norm_conv` | [hidden, branches], effective gamma |
| `conv` | [conv_kernel, hidden * branches], oldest tap first |

The prototype bounds N to 2..16, H to 1..64, embedding_dim to a multiple of 32 up
to 4096 divisible by H, and kernel size to 1..16. Each table is bounded to 1,048,576
rows and 67,108,864 elements. Quantized table rows also need ggml block alignment.
These are fixture limits, not support for the released model's large PLE tables
or native `qwen4exp` GGUF tensor/metadata naming.

One Engine owns one sequence. `decode` advances it once. CUDA `prefill` handles
all supplied tokens layer by layer and exports their KV asynchronously; CPU
`prefill` retains sequential reference execution. Floating-point reductions can
differ across input partitions. `reset` clears PLE/GDN/index state, CPU KV lengths,
sampling history and statistics, and restores weights after failed prefill.
Invalid tokens/capacity are rejected before state changes.
An individual Engine is not thread-safe. `Options.context` / `--context` can
reduce sequence capacity from the GGUF value; zero retains the model capacity.

## Current performance limits

In the default CPU KV mode, decode runs one GPU graph per span between two CPU
attention layers: the previous layer's output projection and FFN, the GDN/PLE
layers in between and the next layer's projections. Q, K and V return as one
packed download; each span synchronizes once because the next CPU attention needs
its result. Spans are launched asynchronously, and while one executes the host
builds and allocates the next from an alternate allocator. Prefill KV exports and
weight prefetch overlap work. Persistent GPU KV decode instead runs one graph per
token on a reused compute buffer and, for Qwen3.5, reuses that graph across tokens
(see [GPU prefill](prefill.md#persistent-gpu-kv-benchmark-mode)).
CPU attention runs on a persistent worker pool shared by all attention layers, and top-k materializes a score for every compressed block. There
is no CPU KV graph reuse, server, multi-sequence cache,
checkpoint converter or long-context benchmark claim. These are explicit next
optimization/integration boundaries after correctness and GPU runtime validation.
