# Benchmarking and profiling

`tidal-bench` measures single-sequence inference and exports machine-readable
results for comparisons. It uses the same `Engine` as `tidal-cli`; CUDA mode
uses GPU layer-major prefill and, by default, CPU attention during decoding.
`--kv-placement gpu` also keeps decode attention and persistent KV on CUDA. Build with the normal
CMake preset. No Python dependency is needed to run the benchmark itself.

## Throughput baseline

Generate a fixture once (the generator refuses to overwrite an existing model):

```sh
mkdir -p models
build/dev-cuda/bin/tidal-dummy --output models/bench-q4.gguf --type q4_0 --context 256
build/dev-cuda/bin/tidal-bench --model models/bench-q4.gguf --backend cuda \
  --prompt 24 --steps 32 --warmup 2 --repetitions 10 --threads 4 \
  --label baseline --json build/dev-cuda/baseline.json
```

Use `--backend cpu` for the CPU implementation. `--prompt N` creates reproducible
token IDs `(17*i + 1) % vocab`; `--tokens 1,7,23,9` supplies a specific sequence
instead. Prompt length plus `--steps` must fit the model's context. Generation
defaults to greedy device sampling; [sampling options](sampling.md) include
temperature, top-k/top-p/min-p, penalties and an explicit CPU path. `--steps 0`
measures prompt ingestion alone; decode metrics are then null.

The terminal and JSON report prefill, decode and total tokens/second separately.
CUDA prefill processes the entire prompt layer by layer using GPU attention;
CPU prefill is the sequential reference. See [prefill controls](prefill.md).
Rates are total processed tokens divided by total elapsed time across measured
repetitions, not the arithmetic mean of per-run rates. The JSON also includes
per-run rates, mean/median/p95/min/max/population-standard-deviation, individual
decode-token latencies, generated tokens, output checksums and engine counters.
Prefill and total token latency fields are null: batch completion is measured as
a whole, not presented as individually measured token latencies.

Timed regions include the Engine call, graph construction/allocation, transfers,
CPU attention, result collection and (during generation) configured token selection.
Model loading/backend initialization, state reset, warmup, checksum calculation and
serialization are excluded. Load and reset times are reported separately.
Each warmup and repetition starts from a reset sequence; measured repetitions
must produce identical outputs. Decode latency samples retain their order.
Prefill timing includes configured KV placement (export completion in CPU KV mode)
and restoration of evicted weights.

Reports also identify persistent KV placement/type, decode attention backend,
prefill execution, working precision, automatic or explicit
chunk/tile sizes and memory settings; comparisons require them to match.
Use `--kv-placement gpu --prefill-precision fp16` with an appropriate `--context`
for a GPU KV comparison. The same options reach text inference through
`tools/qwen35_text.py`; see [GPU KV mode](prefill.md#persistent-gpu-kv-benchmark-mode).

Reports contain model configuration, file size and FNV-1a content fingerprint,
backend/device/CPU, thread count, CUDA API driver/runtime versions, compiler,
build type, ggml revision and build-time Git revision/dirty state. The fingerprint
is an identity check, not a cryptographic integrity guarantee. Rebuild after
source changes; Git identity is refreshed at build time. Use `--label` to identify
experiments, particularly when working with uncommitted changes.

Run performance experiments on an otherwise idle machine, with consistent power
and clock settings, workload, thread count and warmup. Do not build or run other
tests concurrently with measurements used to claim a speedup.

## Detailed profiling

```sh
build/dev-cuda/bin/tidal-bench --model models/bench-q4.gguf --backend cuda \
  --prompt 24 --steps 32 --warmup 2 --repetitions 10 \
  --profile nodes --json build/dev-cuda/profile.json --trace build/dev-cuda/trace.json
```

Modes:

| Mode | Instrumentation |
|---|---|
| `off` (default) | Normal benchmark runs; no detailed events retained |
| `phases` | Separate replay with normal graph execution/fusion; graph build, support checks, allocation/free, compute, transfers, layer/token spans and CPU attention stages |
| `nodes` | Same replay plus individually synchronized ggml graph nodes, with op description, tensor name, output/input shapes and types, token position and layer |

Both profiling modes run **after** the normal throughput repetitions. They replay
the measured sequence's generated tokens, so profiling cannot change the input
workload. JSON reports selection and sampled-decision equality against the normal run.
Full-logit capture is disabled by default. With `--logits 1`, it also reports logit
max absolute/normalized RMS differences; otherwise those error fields are null.
Phase profiling requires bit-identical captured outputs and sampled decisions. Node profiling deliberately disables cross-node fusion;
its numerical differences are reported rather than assumed to be zero.

The CPU stages include BF16 KV append, selected-block expansion, worker launch and
join, and per-query-head QK, stable softmax and weighted-V accumulation. Graph
nodes expose matvecs, normalization, convolution, GDN recurrence, RoPE, activations,
QSA scores/top-k, copies and view/layout operations. Named matrices distinguish
residual gates, attention projections, FFN and the output head.

**Profile event durations are host steady-clock wall times.** The CUDA sampler
uses CUDA events for its contribution to the aggregate `transfer_ms` counter;
see [sampling measurement semantics](sampling.md). Backend compute
calls are synchronized. Node durations include submission/synchronization overhead;
a ggml node may launch several CUDA kernels, while a view node may launch none.
These are not CUDA-event timings, occupancy measurements or hardware memory
bandwidth counters. Use the phase profile for normal graph costs and the node
profile to locate expensive operations. Use Nsight for actual kernel/device metrics.

Totals are inclusive and overlap: token/layer/graph spans contain child events,
and CPU worker heads execute concurrently. Do not sum all categories to obtain
inference time. Worker lanes in the exported trace are logical query-head lanes;
lane 0 is the calling thread. Transfer byte counts are actual API payload sizes;
node bytes are logical output size; CPU QK/V bytes are logical BF16 cache reads,
which can be served from caches and repeat across grouped heads.

`--json` stores every event plus aggregates grouped by category, phase, layer,
name and op, with counts, total/min/max/median/p95 durations and bytes. Raw events
retain shapes and token positions even when the aggregate covers varying context
lengths. The terminal prints the twenty largest non-parent aggregates. `--trace`
exports Chrome Trace Event JSON suitable for Perfetto or a compatible trace viewer.

The default event limit is 500,000. Change it with `--profile-limit N` for larger
runs. Inference continues when the limit is reached, but JSON/trace/terminal output
explicitly reports `dropped_events`; the resulting profile is incomplete. The
comparison tool does not compare aggregates from truncated profiles. Throughput
results remain independent of this diagnostic limit.

## Engine API

Library callers can set `Options::profile` to `ProfileMode::Phases` or
`ProfileMode::Nodes`, then read `Engine::profile_events()` and
`Engine::profile_dropped_events()` between decode calls. `set_profile()` changes
instrumentation and clears events without resetting the sequence; `reset()` clears
both sequence state and events. `Options::nvtx` and `profile_event_limit` control
NVTX emission and retention. As with decoding, access to an Engine is not thread-safe.

## Nsight / NVTX

CUDA builds enable header-only NVTX support when CUDA's `nvtx3` headers are found.
Add `--nvtx 1` to emit named ranges for tokens, layers, graph work, transfers and
CPU attention. Labels include phase, layer and token position. `--profile off`
with NVTX retains normal graph execution for external CUDA profiling; `nodes`
also labels each individually executed ggml operation in its separate replay.
A build without NVTX support rejects the request explicitly.

For example, with Nsight Compute and permission to access GPU profiling counters:

```sh
ncu --nvtx --section LaunchStats --launch-count 1 \
  --export build/dev-cuda/launch-profile \
  build/dev-cuda/bin/tidal-bench --model models/bench-q4.gguf --backend cuda \
  --prompt 8 --steps 4 --warmup 0 --repetitions 1 --nvtx 1
```

The same NVTX ranges are available to Nsight Systems when installed. Throughput
measured under an external profiler is affected by that profiler's overhead and
must not be used as the ordinary performance baseline.

If Nsight Compute reports `ERR_NVGPUCTRPERM`, the host NVIDIA profiling-counter
permissions need configuration before hardware metrics can be collected. Built-in
wall-time profiling does not depend on those counters.

## Improvement / regression comparison

After rebuilding the candidate, run the same benchmark arguments with
`--label candidate --json build/dev-cuda/candidate.json`, then:

```sh
python3 tools/compare_bench.py build/dev-cuda/baseline.json build/dev-cuda/candidate.json \
  --max-regression 5 --json build/dev-cuda/comparison.json
```

The tool reports throughput percentage changes and p95 latency. Compatible,
complete profiles additionally report the largest changes in inclusive stage
cost. It rejects mismatched model/workload/backend/device/CPU/thread/CUDA-version/
NVTX/warmup/sampling/capture settings or different generated tokens.
`--allow-sampler-change` permits comparing CPU and device sampling with all
other settings held constant. Repetition counts can differ.
Different output checksums are reported for numerical follow-up; performance
comparison does not replace `test_reference.py`.

Exit status is 0 for success, 1 when any prefill/decode/total throughput falls more
than the requested percentage, and 2 for incompatible or invalid reports. Without
`--max-regression`, comparison is informational. A single noisy run is not evidence
of a regression; use repeated measurements and inspect their dispersion.

## Verification

```sh
ctest --preset dev-cuda
python3 tests/test_benchmark.py --bin-dir build/dev-cuda/bin --backend cuda
python3 tests/test_reference.py --bin-dir build/dev-cuda/bin --backend cuda
```

The benchmark test checks accounting, absence of warmup/repetition leakage in the
profile replay, output preservation, per-node coverage, CPU head stages, transfer
byte totals, trace JSON, truncation signaling, zero-generation behavior, invalid
inputs and deterministic regression-threshold handling. CTest runs these checks on
CPU; the explicit CUDA command verifies the real split path.

## Local result privacy

Benchmark reports can fingerprint a machine: they contain hardware and software
metadata, model identities, paths, timings and generated tokens. Keep them local
unless you deliberately choose to share them. The benchmark and comparison tools
write local files and do not upload reports or send telemetry.

The documented output paths are under Git-ignored `build/` directories. Model
files, `/benchmarks/`, `*.bench.json`, `*.trace.json`, and native Nsight reports
are also ignored. An arbitrary `--json` path outside those patterns is not
automatically excluded; check it with `git check-ignore -v PATH` before committing.
Only reproducible test contracts and usage examples belong in repository docs,
not observations from a particular local machine.
