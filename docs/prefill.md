# Layer-major GPU prefill

CUDA `Engine::prefill` processes every supplied prompt token through one decoder
layer before advancing to the next. The order is embedding, then each layer's
optional PLE, GDN or attention, and FFN, followed by the output head. Query tiles
stay inside the current attention layer; they never traverse the whole model.
The explicit CPU backend retains sequential reference execution.

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

The canonical KV representation remains BF16. New KV is produced on GPU and
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

## Controls and measurements

The CLI and benchmark accept:

- `--prefill-tile N`: attention queries per tile, default 128; range 1–4096.
- `--prefill-precision fp16|bf16`: GPU attention working KV format.
- `--prefill-memory-mib N`: optional managed weight/workspace cap for pressure
  experiments; zero uses live availability.
- `--prefill-reserve-mib N`: device headroom, default 64 MiB.

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

See [validation](validation.md) for reproducible correctness checks. Models,
measurement JSON, profiler traces and local development explanations belong in
ignored `models/` or `build/` directories.
