# Synthetic 27B Dense integration fixture

`--preset dense-27b` constructs a hypothetical Qwen-derived Dense model with
27,108,415,232 real stored parameters. It is random and untrained, not a released
Qwen checkpoint. PLE and MoE are absent. The preset keeps the GDN/QSA rhythm,
hyper-connections, normalization, RoPE, gated attention and Dense SwiGLU.

The preset initializes FFN-down, GDN-output and attention-output weights with an
additional `1 / layers` scale, recorded as `tidal.residual_init_scale`. This keeps
an untrained deep residual network well conditioned for numerical comparisons:
unscaled random recurrent networks can amplify small backend activation-quantizer
differences across layers and tokens. All weights remain nonzero random tensors;
the inference equations are unchanged. This is a synthetic initialization choice,
not a quantization-quality claim for a trained checkpoint.

| Dimension | Value |
|---|---:|
| Hidden / FFN | 5120 / 18944 |
| Layers / residual branches / low-rank width | 64 / 4 / 256 |
| Vocabulary / context capacity | 131072 / 2048 |
| Q / KV heads, head width | 40 / 8, 128 |
| GDN key / value heads, state width | 8 / 40, 128 |
| QSA index heads / width | 4 / 128 |
| Index block size / selected block budget | 16 / 8 |

`configs/dense-27b-6bpw.types` assigns Q5_K to FFN gate/up matrices, Q8_0 to token
embeddings and retains the command's Q6_K default for other matrices. Norm,
convolution and decay parameters stay F32. The deterministic payload is
20,763,849,728 bytes, about **6.128 bits per parameter**, before small GGUF metadata
and alignment overhead. This leaves space within a 24-GiB device for state and
workspace, but successful runtime allocation is still checked explicitly.

## Generate and test

```sh
mkdir -p models build/large-validation
build/dev-cuda/bin/tidal-dummy --preset dense-27b --type q6_K \
  --tensor-types configs/dense-27b-6bpw.types --dry-run 1
build/dev-cuda/bin/tidal-dummy --preset dense-27b --type q6_K \
  --tensor-types configs/dense-27b-6bpw.types --threads 4 --chunk-mib 16 \
  --output models/qwen4exp-dense-27b-6bpw.gguf
python3 tests/test_large.py --bin-dir build/dev-cuda/bin \
  --model models/qwen4exp-dense-27b-6bpw.gguf
```

The generator writes every parameter; this is not a sparse-file size placeholder.
It needs roughly 21 GB of free disk. Generation and loading use bounded staging
buffers instead of keeping an F32 copy of the entire model. CPU comparison also
needs enough RAM for the roughly 21-GB encoded model plus state and workspace.
The optional test is deliberately excluded from routine CTest/CI.

The integration test verifies the parameter count, mixed formats, PLE/MoE absence,
native weight residency with no load-time widening, CUDA generation and exact
reset/replay. It also checks CPU attention calls, BF16 KV placement and every
CPU/GPU transfer byte against the architecture's expected shapes. It samples the
process's VRAM usage and checks it against 24 GiB.
Run without starting other GPU jobs: inside a PID namespace the sampler matches
the sole new NVIDIA process and rejects ambiguous concurrent processes.
Sixteen prompt tokens complete an index block; two generated tokens also exercise
the incomplete tail. The same tokens are teacher-forced through CPU inference.
Full-vocabulary logits require normalized RMS difference below 5% and cosine
similarity above 0.995; QSA selections must match. CPU/CUDA quantizers differ, so
this is a backend comparison, not an independent trained-model correctness proof.
Fine-grained equation tests remain the small independent-oracle suites.

Models (`*.gguf`), incomplete outputs (`*.gguf.partial`), and `build/` are ignored.
Raw logits, process logs, sampled memory and timing reports stay under
`build/large-validation/`. Local results and machine fingerprints are not stored
in this document or commits. No script uploads or pushes them.
