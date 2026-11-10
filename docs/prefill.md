# Layer-major GPU prefill

CUDA `Engine::prefill` processes every supplied prompt token through one decoder
layer before advancing to the next. The order is embedding, then each layer's
optional PLE, GDN or attention, and FFN, followed by the output head. Query tiles
stay inside the current attention layer; they never traverse the whole model.
The explicit CPU backend retains sequential reference execution.

Within a layer, embedding, PLE, GDN, Dense/MoE FFN, attention projections/KV layout
and diagnostic heads use bounded token ranges. GDN and PLE carry their recurrent
and convolution histories between ranges. Attention retains the active layer's
full KV and a range of queries/gates/output; it never materializes those three
intermediates for the entire prompt. The layer's hidden activation buffer remains
resident until every layer has finished. A range does not advance to the next
decoder layer before the rest of the prompt.

Native `qwen35` uses full causal attention. Synthetic `qwen4exp` Dense/MoE uses
QSA: each query can rank only blocks complete at its own position, and always
includes its incomplete tail. Pooling later blocks cannot expose future tokens.
GDN recurrent/convolution state, PLE histories and residual branches are retained
for subsequent prefill calls or single-token decoding.

## Attention and KV lifetime

Projections, QSA scoring/top-k, causal masks, attention and output projections
execute on CUDA. Attention uses ggml FlashAttention with bounded query tiles;
there is no CPU attention fallback. Unsupported head widths are zero-padded to a
supported width while retaining the original attention scale.

With the default `--kv-placement cpu`, canonical KV remains BF16. New KV is produced on GPU and
exported through a separate CUDA stream using pinned host staging. The transfer
owns its source allocation until completion. A bounded queue prevents outstanding
copies from retaining an unbounded number of GPU caches. Before prefill returns,
all KV exports are complete and the CPU cache is ready for decoding. Extending an
existing sequence first uploads that layer's CPU KV prefix.

`--prefill-precision fp16` (default) uses FP16 working KV for GPU attention,
including the Tensor Core path on supported older CUDA architectures. `bf16`
keeps BF16 working KV and lets the pinned ggml backend choose supported arithmetic;
it does not promise native BF16 Tensor Cores on older devices. Norms, residuals,
GDN state and graph activations remain F32; model weights retain their per-tensor
GGUF types and use ggml's native matrix kernels. Operand conversion and reduction
order can differ from the CPU-attention decoder.

## Persistent GPU KV benchmark mode

`--kv-placement gpu` is an opt-in CUDA mode for comparing GPU execution with
other engines. The default `cpu` preserves hybrid inference. GPU mode allocates
all attention layers' KV in device memory at initialization, sized by `--context`
and padded to the FlashAttention head/token alignment. `--prefill-precision`
selects the persistent format for **prefill and decode**: FP16 by default, or
BF16. FP16 keys/values are stored directly from projections, without an
intermediate BF16 round trip. Compressed QSA index keys remain BF16 in both modes.

Prefill appends directly to these buffers, with no temporary full-cache replica,
CPU KV allocation, KV export or prefix upload. Decode appends the next token and
runs full attention or causal QSA on CUDA. The QSA mask includes ranked complete
blocks and the incomplete tail; its finite-entry bound allows ggml to select a
sparse-gather kernel when supported. Prefix extension, decode-to-prefill and reset
use the same persistent cache. No implicit switch to CPU occurs on exhaustion.

Because no decode work leaves the device in this mode, a compact decode step
(`decode(token, false)`) builds a single graph from the embedding to the logits,
allocates it from a compute buffer kept across tokens and synchronizes once. For
Qwen3.5 the token ID, RoPE position, KV destination row and an F16 causal mask
shared by all attention layers are graph inputs, so the graph is reused until
the 256-token KV bucket grows, a prefill runs (it may reload weights) or the
engine is reset; ggml may then replay it as a CUDA graph. `decode_graph_reuses`
counts reused tokens. QSA and PLE graphs contain position-dependent structure
and are rebuilt per token. Decode that captures logits keeps the per-layer
graphs used for diagnostics.

For example:

```sh
build/dev-cuda/bin/tidal-bench --model models/model.gguf --backend cuda \
  --kv-placement gpu --prefill-precision fp16 --context 4096 \
  --prompt 2048 --steps 32 --json build/gpu-kv.bench.json
```

The complete model, persistent GPU KV/state and decode workspace must fit at
once. Automatic prefill chunks only bound temporary work; they cannot reduce the
allocated KV capacity. Oversized KV/state fails with an allocation error that
suggests reducing `--context` or explicitly selecting CPU KV. Prefill can still
evict weights for temporary workspace and restores them before returning.
The managed prefill cap covers weights and temporary workspace; live device
availability additionally accounts for persistent KV/state and kernel memory.
GPU KV with the explicit CPU backend is rejected.

Reports identify KV placement/type and the decode attention backend.
`gpu_kv_bytes` reports persistent padded K/V storage, `gpu_attention_calls`
counts decode attention calls, and `prefill_gpu_attention_calls` counts prefill
tiles. CPU attention counters and `prefill_kv_to_host_bytes` remain zero in GPU
mode. Disabling diagnostics keeps Q/K/V, attention outputs and selected IDs on
device; PLE hashing, graph orchestration and tokenization still use the host.

This mode matches the KV placement/precision dimensions of a GPU baseline.
Batch scheduling, weight placement, ggml version and kernel choices must also
be recorded when comparing engines; identical FP16 KV does not imply bitwise
identical logits or identical execution algorithms.

## Weight residency

Weights have independently allocated embedding, attention, FFN, PLE and output
head groups. Tied embeddings have one allocation. The original GGUF handle and
tensor offsets are retained for reloads, with bounded pinned staging and a
separate upload stream. The file must remain readable and its contents unchanged
while the engine is alive. Reloads can involve storage I/O if the OS file cache
no longer holds the corresponding data.

Before allocating graph or persistent prefill workspace, the planner consults
live device availability and a configurable reserve. When space is insufficient:

1. Evict unprotected groups in already completed layers.
2. Evict future layers, farthest first.
3. If still needed, evict inactive sublayers of the current layer.

The active sublayer's weights remain protected. Upcoming groups are prefetched
asynchronously when space permits, and execution waits before using them.
Completed-layer weights are restored only after all prefill graphs and KV exports
finish and temporary prefill storage is released. Decode weights initially need
to fit the device, and that layout is restored before returning.

If protected weights plus required workspace cannot fit after reclamation and
eligible evictions, prefill throws an insufficient-VRAM error. A failed prefill
poisons the partially updated sequence; call `reset()` before reuse. There is no
implicit CPU placement. ggml kernel-internal allocations also consume VRAM;
`--prefill-reserve-mib` leaves headroom for them and other device users.

Automatic chunk sizing estimates temporary bytes per token from model dimensions
and the current phase, and uses at most one quarter of the memory recoverable by
evicting unprotected weights. It rounds down to a power of two to retain wide
matrix operations. Actual ggml graph sizing remains the allocation gate: if a
pre-compute allocation fails, the uncommitted range is halved and retried, down to
one token. Later successful ranges use the reduced width. A new phase recomputes
its initial estimate against live availability. Attention query tiles follow the
same policy with a 4,096-query ceiling and estimates that grow with KV length.

Only allocation failures before graph execution are retryable. A later compute
error or a failure in a consumer phase poisons the sequence without replaying
already committed recurrent/QSA state. Failure at one token reports the phase and
position. Explicit sizes fail instead of silently changing the requested setting.
The irreducible hidden buffer, active-layer KV, protected weights and CPU KV must
still fit their respective memories; shrinking temporary ranges cannot eliminate
these allocations. Diagnostic capture also retains per-position host results.

Default kernel/runtime headroom is automatic: the larger of 64 MiB and 1/16 of
device capacity. This is separate from the managed graph budget. It protects
against typical ggml kernel-pool allocations, but cannot promise recovery from
every external allocation race or a backend that aborts internally. A numeric
reserve overrides the heuristic; zero disables the reserve.
After temporary storage and the prefill kernel pool are released, restoring the
original decode weights uses actual free device memory rather than retaining the
prefill-only reserve.

## Controls and measurements

The CLI and benchmark accept:

- `--prefill-chunk auto|N`: temporary tensor/GEMM token range, default `auto`.
  Explicit `N` must be positive; the final range can be shorter.
- `--prefill-tile auto|N`: attention query tile, default `auto`; explicit 1–4096.
- `--kv-placement cpu|gpu`: persistent KV placement and decode attention backend;
  default `cpu`. GPU placement requires CUDA.
- `--prefill-precision fp16|bf16`: GPU attention working KV format, also used for
  persistent KV and decoding with `--kv-placement gpu`.
- `--prefill-memory-mib N`: optional managed weight/workspace cap for pressure
  experiments; zero uses live availability.
- `--prefill-reserve-mib auto|N`: device headroom, default `auto`.

For example, use `--prefill-chunk 4096 --prefill-tile 256` to compare fixed
partitions, or omit both flags to adapt to the model and available memory. The
C++ equivalents are `Options::prefill_chunk_tokens` and
`Options::prefill_attention_tile` (zero means automatic);
`Options::prefill_memory_reserve == UINT64_MAX` means automatic headroom.

Compact inference returns prompt positions without vocabulary vectors or
per-query selection lists. Diagnostic capture returns both for every position;
large prompts can therefore consume substantial host memory. The final sampling
head uses the same arithmetic in either mode.

The benchmark times the complete prefill API, including KV completion and weight
restoration. Prefill throughput uses the entire prompt count and elapsed API time.
There are no fabricated per-token latency samples for this batch operation;
`token_latency_ms` is null for prefill and total, and remains measured for decode.
Saved reports identify execution mode, working precision, tile and memory settings;
the comparison tool rejects mismatched settings.
The CLI's legacy `attention_backend` and `sparse_attention_backend` fields describe
decoding; the separate `prefill` object identifies prompt execution and placement.

Profile phases distinguish embedding, PLE, GDN, attention preparation/layout/
FlashAttention/output, FFN, head, KV export and weight residency. Counters report
prefill tokens/layers/attention tiles, exported KV bytes, weight evictions,
reloads/prefetches and peak managed workspace. Host wall times from asynchronous
transfers can overlap compute; adding them does not produce elapsed prefill time.
Reports also identify automatic versus explicit chunk/tile/reserve settings and
record actual minimum/maximum range sizes (including short tails), completed
range count, allocation retry count and resolved headroom. `prefill/chunk` profile
events carry the range length in `shape`; `memory/chunk_reduce` carries the failed
and reduced lengths. Failed attempts do not increment completed-range counters.

See [validation](validation.md) for reproducible correctness checks. Models,
measurement JSON, profiler traces and local development explanations belong in
ignored `models/` or `build/` directories.
