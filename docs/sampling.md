# Sampling

Generation defaults to sampling on the model's backend (`--sampler device`): CUDA
for a CUDA model, CPU for a CPU model. `--sampler cpu` explicitly downloads logits
and uses the CPU implementation while leaving the neural network on its selected
backend. Unsupported device sampling raises an error; it never silently falls
back. The existing greedy behavior is preserved by the default temperature of 0.

```sh
build/dev-cuda/bin/tidal-cli --model models/model.gguf --backend cuda \
  --tokens 1,2,3 --steps 32 --temp 0.8 --top-k 40 --top-p 0.95 \
  --min-p 0.05 --repeat-penalty 1.1 --seed 42
```

The CLI, benchmark and optional `tools/qwen35_text.py` frontend share these options:

| Option | Default | Meaning |
|---|---|---|
| `--sampler` | `device` | Model device or explicit `cpu` |
| `--temp` | `0` | 0: greedy; positive: temperature sampling |
| `--top-k` | `0` | Keep highest k; 0 or -1 disables; oversized k clamps to vocabulary |
| `--top-p` | `1` | Keep smallest sorted prefix whose cumulative probability reaches p; 0 < p <= 1 |
| `--min-p` | `0` | Keep probabilities at least this fraction of the maximum; 0 <= p <= 1 |
| `--repeat-penalty` | `1` | Divide positive seen logits, multiply nonpositive seen logits; must be positive |
| `--frequency-penalty` | `0` | Subtract value times recent occurrence count |
| `--presence-penalty` | `0` | Subtract value once from each recently seen token |
| `--penalty-last-n` | `64` | Consumed-token window; -1: all; 0: disable penalties |
| `--seed` | `0` | Unsigned 64-bit seed; reset restores RNG state |

Penalties apply as repeat, presence, then frequency; followed by temperature,
top-k, min-p and top-p, then categorical selection. Top-p is normalized after top-k/min-p filtering.
Greedy applies penalties but ignores temperature-dependent filters. Equal scores
use ascending token ID, including at a top-k boundary. Negative frequency/presence
penalties encourage seen tokens. Nonfinite parameters/logits and penalty overflow
are rejected. NaN/Inf validation also runs on unsampled prompt positions.

History contains consumed prompt and generated tokens. Calling `sample()` draws
without consuming a token; pass the result to `decode()` to consume it. Prefill
never advances RNG. A single host SplitMix64 state supplies each probabilistic
draw's uniform variate to either implementation; GPU filtering and token selection
remain on the GPU. Reset replays the same sequence on the same backend. Floating
point reductions and exponentials can differ across devices, so identical seeds
do not promise identical tokens for distributions at a rounding boundary.

`Engine::set_sampling(config)` replaces the parameters of a live engine, between
prompts or mid-sequence: it restarts the random sequence from the new seed and
rebuilds the penalty history from the consumed tokens, so the next draw is the
one a fresh engine with that configuration would make after the same sequence.
The sampler backend stays the one chosen at construction. `Engine::logits()`
returns the last position's unpenalized logits (one vocabulary-sized
device-to-host copy on CUDA). `tidal-server` uses both: per-request sampling
parameters and token probabilities.

## API and logit capture

Set `Options::sampling` before constructing an engine. Call
`prefill(tokens, false)`, then `sample()` and `decode(token, false)` for generation.
`sample()` before any decoded token is an error. Execution failures require
`reset()` before reuse. Existing `decode(token)` and `prefill(tokens)` callers
retain full-logit capture for numerical validation.

The CLI defaults to no full-logit readback unless `--json` is requested, preserving
the diagnostic JSON contract. `--logits 0` overrides that behavior; the JSON still
contains token IDs, selections and statistics, with empty logit arrays. The text
frontend always uses this compact path. Benchmarks default to `--logits 0`, even
with a JSON report; use `--logits 1` explicitly for logit comparisons.

The head retains one F32 vocabulary vector on the model backend. Compact decoding
reads a four-byte finite-status reduction per position. CUDA sampling downloads a
12-byte token/error/candidate-count result per draw. Stochastic draws upload one
eight-byte uniform variate; enabled penalties upload only changed token/count
pairs. Explicit CPU sampling downloads a full F32 vocabulary vector per draw,
unless diagnostic capture already supplied it. Neural-network/CPU-attention
transfers are separate and unchanged in purpose.

## Implementation and portability

`src/sampling.cpp` contains the CPU reference, policy validation, history and RNG.
`src/sampling_device.hpp` defines a narrow device interface. `src/sampling_cuda.cu`
implements stable greedy reduction, CUB stable radix sort, double-precision
probabilities, a deterministic hierarchical prefix sum and device categorical
selection. The fixed scan topology avoids scheduling-dependent rounding from
parallel lookback scans. Scratch buffers are
allocated once per engine; the private stream synchronizes before results are
used. Model graph completion precedes sampling. This is a correctness-oriented
implementation: probabilistic sampling currently sorts the full vocabulary even
when k is small. It does not implement vLLM's fused rejection-sampling algorithms.

HIP, SYCL and Vulkan sampling implementations are not provided. Their future
implementations can reuse policy, history, RNG and the independent distribution
tests. CUDA pointers, streams and CUB stay inside the CUDA implementation; adding a
new device requires adapting the tensor/storage interface as well as its kernels.
CPU sampling remains usable for bring-up and comparison. Grammar constraints,
logit bias and speculative/batched sampling are outside this implementation.

## Measurement and validation

`sampling_calls`, `sampling_ms`, `sampling_device_to_host_bytes` and
`sampling_host_to_device_bytes` are reported separately. Sampling wall time is
inclusive of transfers and overlaps transfer accounting. CUDA sampler copy time
in `transfer_ms` uses CUDA events; existing engine/CPU transfer timing is host
wall time, so that aggregate is not a single device-time measurement. Detailed
profile event durations remain synchronized host wall times. Profiling exposes
penalties, sorting, temperature/min-p, cumulative sum, top-p/draw and transfers.
NVTX or detailed profiling synchronizes these stages and can affect latency.

For a controlled CPU/GPU sampler comparison, keep `--backend cuda`, model,
prompt, sampling parameters, seed, capture mode and warmup identical. Change only
`--sampler device` versus `--sampler cpu`, store reports under `build/`, then run:

```sh
python3 tools/compare_bench.py build/device.json build/cpu.json --allow-sampler-change
```

The comparison still rejects different parameters or generated sequences. The
normal measurement includes sampling; profile replay uses the same capture mode
and fixed consumed tokens. Logit error fields are null when capture is disabled.
Checksums then cover positions, selections and generated tokens, not logits.

```sh
python3 tests/test_sampling.py --bin-dir build/dev-cuda/bin --backend cuda
```

The suite checks an independent NumPy distribution oracle, ties, top-p boundaries,
large vocabularies, extreme finite logits, NaN/Inf rejection, penalty windows,
negative penalties, seeded categorical frequencies, resets, end-to-end generation,
compact/full-logit equivalence, transfer payloads and profiling. CTest registers
the CPU version when NumPy is available. Local reports and models stay ignored.
