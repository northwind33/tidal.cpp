#!/usr/bin/env python3
"""Run the actual binaries and compare full trajectories to an independent oracle."""
import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

# Tiny matrices are much faster with one BLAS thread, independently of ggml threads.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from reference import Reference, read_gguf


FORMATS = ["f32", "f16", "bf16", "q8_0", "q4_0"]
PROMPT = [1, 7, 23, 9, 42, 5, 6, 2, 77, 13, 84, 11, 90, 3, 42, 28, 61, 7, 0, 93, 34, 22, 1, 255]
TOLERANCES = {
    "cpu": dict(zip(FORMATS, [2e-4, 3e-3, 1.2e-2, 2.5e-2, 2.5e-2])),
    # F32 differences can cross BF16 cache rounding boundaries. CUDA floating
    # matvecs keep F32 inputs, so F16/BF16 need much tighter bounds than on CPU.
    "cuda": dict(zip(FORMATS, [4e-4, 1e-3, 4e-4, 2.5e-2, 2.5e-2])),
}


def run(*args, ok=True, input=None):
    result = subprocess.run([str(a) for a in args], input=input, capture_output=True, text=True, timeout=120)
    if (result.returncode == 0) != ok:
        raise AssertionError(f"{' '.join(map(str, args))}\n{result.stdout}\n{result.stderr}")
    # A rejected CLI input should be a catchable error, never a ggml assertion/abort.
    if not ok:
        assert result.returncode == 1, result.stderr
    return result


def check_matvecs(binary, model, backend):
    reference = Reference(model, backend)
    rng = np.random.default_rng(20261104)
    requests, expected = [], []
    # Include short and long rows, small outputs, and all input quantization
    # edge cases independently of the nonlinear/recurrent model trajectory.
    names = ["blk.0.hc_attn.down", "blk.0.hc_attn.inject", "blk.0.hc_attn.up",
             "blk.0.gdn.qkv", "output.weight"]
    names += [f"blk.{layer}.ple.{part}" for layer in reference.c.get("ple.layers", []) for part in ["key", "value"]]
    for name in names:
        width = reference.w[name].shape[1]
        mixed = rng.uniform(-2, 2, width).astype(np.float32)
        if width > 32:
            mixed[:32] = 0  # Zero activation block must not divide by zero.
        below_half = np.nextafter(np.float32(0.5), np.float32(0))
        ties = np.resize(np.array([127, -127, 0.5, -0.5, 1.5, -1.5, below_half, -below_half],
                                  dtype=np.float32), width) / np.float32(128)
        # Power-of-two scaling preserves Q8 rounding boundaries while avoiding
        # large F32 cancellation errors in the floating-point matvec fixtures.
        inputs = [np.zeros(width, dtype=np.float32), mixed, ties]
        if reference.types[name] in (2, 8):
            inputs.append(np.resize(np.arange(-8, 8, dtype=np.float32), width) * np.float32(np.float16(.012)))
        for x in inputs:
            requests.append(name + " " + " ".join(map(str, x)))
            expected.append((name, reference.mm(name, x)))
    for layer in reference.c.get("ple.layers", []):
        name = f"blk.{layer}.ple.embedding"
        rows = reference.c[f"ple.{layer}.offsets"]
        rows = [rows[-1], 0, *rows, reference.w[name].shape[0] - 1]
        requests.append("@rows " + name + " " + str(len(rows)) + " " + " ".join(map(str, rows)))
        expected.append((name, reference.w[name][rows].reshape(-1)))
    result = run(binary, model, backend, input="\n".join(requests) + "\n")
    lines = result.stdout.splitlines()
    assert len(lines) == len(expected), result.stdout
    for line, (name, values) in zip(lines, expected):
        fields = line.split()
        assert fields[0] == name
        np.testing.assert_allclose(np.array(fields[1:], dtype=np.float32), values,
                                   atol=2e-5, rtol=2e-5, err_msg=f"{backend} matvec {name}")


def check_placement(result, config, backend, steps):
    assert result["backend"].startswith("CUDA" if backend == "cuda" else "CPU")
    assert result["sparse_attention_backend"] == "CPU"
    assert result["kv_type"] == "BF16"
    stats = result["statistics"]
    qsa_layers = config["layers"] // 4
    assert stats["sparse_attention_calls"] == steps * qsa_layers
    assert stats["backend_graph_nodes"] > 0
    # Only current Q/K/V, selected IDs and final logits leave the device.
    # Account for every byte, including RoPE positions and token IDs on upload.
    q_width = config["q_heads"] * config["head_dim"]
    kv_width = config["kv_heads"] * config["head_dim"]
    ids = sum(min(config["top_blocks"], visible // config["block_size"])
              for visible in range(1, steps + 1))
    d2h = 4 * (steps * config["vocab"] + qsa_layers * (steps * (q_width + 2 * kv_width) + ids))
    h2d = 4 * (steps + qsa_layers * (steps * (3 + q_width) + steps // config["block_size"]))
    if config.get("ple.layers"):
        h2d += 4 * steps * len(config["ple.layers"]) * (config["ple.ngram_size"] - 1) * config["ple.heads_per_ngram"]
    assert stats["device_to_host_bytes"] == (d2h if backend == "cuda" else 0), stats
    assert stats["host_to_device_bytes"] == (h2d if backend == "cuda" else 0), stats


def check_trace(result, reference, tolerance, label, prompt=PROMPT):
    sequence = prompt + result["generated"]
    assert len(result["trace"]) == len(sequence)
    max_error, max_nrmse = 0, 0
    for i, token in enumerate(sequence):
        expected, selected = reference.decode(token)
        step = result["trace"][i]
        actual = np.array(step["logits"])
        max_error = max(max_error, float(np.max(np.abs(expected - actual))))
        np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance,
                                   err_msg=f"{label}, token position {i}")
        nrmse = float(np.sqrt(np.mean((actual - expected) ** 2) / np.mean(expected ** 2)))
        max_nrmse = max(max_nrmse, nrmse)
        assert nrmse < 0.025, (label, i, nrmse)
        assert step["position"] == i
        assert step["selections"] == selected, (label, i, step["selections"], selected)
        if len(prompt) - 1 <= i < len(sequence) - 1:
            # Check the oracle's greedy decision, not just logits conditioned
            # on tokens chosen by the engine under test.
            assert sequence[i + 1] == int(np.argmax(expected)), (label, i, "greedy token")
    return max_error, max_nrmse


def check_format(root, bins, backend, kind):
    model, trace = root / f"{kind}.gguf", root / f"{kind}.json"
    run(bins / "tidal-dummy", "--output", model, "--type", kind, "--context", 32)
    check_matvecs(bins / "tidal-matvec-test", model, backend)
    run(bins / "tidal-cli", "--model", model, "--backend", backend,
        "--tokens", ",".join(map(str, PROMPT)), "--steps", 6, "--json", trace,
        "--repeat", 2, "--threads", 2)
    result, reference = json.loads(trace.read_text()), Reference(model, backend)
    assert len(result["generated"]) == 6
    check_placement(result, reference.c, backend, 30)
    error, nrmse = check_trace(result, reference, TOLERANCES[backend][kind], kind)
    print(f"{kind}: matvecs + 30-token logits + QSA + greedy generation + reset passed; "
          f"max_abs={error:.8g}, max_nrmse={nrmse:.8g}", flush=True)


def check_multi(root, bins, backend):
    # A second superblock catches per-layer cache aliasing.
    model, trace = root / "multi.gguf", root / "multi.json"
    run(bins / "tidal-dummy", "--output", model, "--layers", 8, "--context", 32,
        "--seed", 123, "--hidden", 256)
    run(bins / "tidal-cli", "--model", model, "--backend", backend,
        "--tokens", ",".join(map(str, PROMPT)), "--steps", 0, "--json", trace, "--repeat", 2)
    result, reference = json.loads(trace.read_text()), Reference(model, backend)
    assert result["generated"] == []
    check_placement(result, reference.c, backend, len(PROMPT))
    check_trace(result, reference, 3e-4, "8 layers / hidden 256")
    print("8 layers / hidden 256: two independent QSA caches + reset passed", flush=True)


def check_wide_heads(root, bins, backend):
    # Retain the 27B preset's 40 heads / width-128 GDN and QSA, while shrinking
    # the residual/FFN/layer count enough for an independent F32 NumPy oracle.
    model, trace = root / "wide-heads.gguf", root / "wide-heads.json"
    run(bins / "tidal-dummy", "--preset", "dense-27b", "--hidden", 256, "--ffn", 512,
        "--layers", 4, "--vocab", 256, "--context", 32, "--output", model)
    run(bins / "tidal-cli", "--model", model, "--backend", backend,
        "--tokens", ",".join(map(str, PROMPT)), "--steps", 0, "--json", trace, "--repeat", 2)
    result, ref = json.loads(trace.read_text()), Reference(model, backend)
    assert ref.c["residual_init_scale"] == .25
    check_trace(result, ref, TOLERANCES[backend]["f32"], "width-128 / 40-head state")
    check_placement(result, ref.c, backend, len(PROMPT))
    print("width-128 / 40-head state: independent logits, QSA and reset passed", flush=True)


def check_ple(root, bins, backend, kind, variant="default"):
    model, trace = root / f"ple-{kind}-{variant}.gguf", root / f"ple-{kind}-{variant}.json"
    extra = ["--ple-layers", "0,2", "--ple-type", kind]
    prompt = [0, 0] + PROMPT[:-2]  # Beginning padding, consecutive EOS, and later EOS.
    if variant == "alternate":
        extra = ["--layers", 8, "--ple-layers", "1,4,6", "--ple-ngram", 4,
                 "--ple-heads", 2, "--ple-dim", 192, "--ple-conv", 3, "--ple-eos", 7]
        prompt = [7, 7] + PROMPT[:-2]
    elif variant == "pointwise":
        extra = ["--ple-layers", "1", "--ple-ngram", 2, "--ple-heads", 1,
                 "--ple-dim", 96, "--ple-conv", 1, "--ple-eos", 255]
    # Keep quantized table lookup under the tight F32 end-to-end contract.
    # Fully quantized models have separate native-kernel and state checks below:
    # small rounding changes at activation-quantizer thresholds can change QSA
    # top-k/argmax, making independent long-trajectory equality inappropriate.
    matrix_kind = "f32" if kind in ("q4_0", "q8_0") and variant == "default" else kind
    run(bins / "tidal-dummy", "--output", model, "--type", matrix_kind, "--context", 32, *extra)
    check_matvecs(bins / "tidal-matvec-test", model, backend)
    run(bins / "tidal-cli", "--model", model, "--backend", backend,
        "--tokens", ",".join(map(str, prompt)), "--steps", 6, "--json", trace, "--repeat", 2)
    result, reference = json.loads(trace.read_text()), Reference(model, backend)
    if variant == "default":
        # Golden constants from the pinned Transformers hash initialization,
        # using seed=42, vocabulary=256 and PLE module ordinals 0 and 1.
        assert reference.c["ple.0.multipliers"] == [26958270162658125, 10360346520797407, 16259569841195497]
        assert reference.c["ple.2.multipliers"] == [3558921994396713, 21786721718226283, 26872783636338227]
        assert reference.c["ple.0.vocab_sizes"] == [17, 19, 23, 29]
        assert reference.c["ple.2.vocab_sizes"] == [31, 37, 41, 43]
    check_placement(result, reference.c, backend, 30)
    if variant != "quantized":
        check_trace(result, reference, TOLERANCES[backend][matrix_kind], f"PLE {kind} {variant}", prompt)
    else:
        assert len(result["trace"]) == 30 and len(result["generated"]) == 6
        assert all(np.isfinite(step["logits"]).all() for step in result["trace"])
    run(bins / "tidal-state-test", model, backend)
    coverage = "oracle logits + QSA + greedy" if variant != "quantized" else "native projection/lookup + finite generation"
    print(f"PLE {kind} {variant}: {coverage} + EOS + reset/chunked state + placement passed", flush=True)


def check_ple_rejections(root, bins, backend):
    model, bad = root / "ple-valid.gguf", root / "ple-bad.gguf"
    run(bins / "tidal-dummy", "--output", model, "--ple-layers", "0,2")
    baseline = root / "ple-free.gguf"
    run(bins / "tidal-dummy", "--output", baseline)
    base_config, base_weights, _, _ = read_gguf(baseline)
    _, ple_weights, _, _ = read_gguf(model)
    assert not any(key.startswith("ple.") for key in base_config)
    for name, weight in base_weights.items():
        np.testing.assert_array_equal(ple_weights[name], weight, err_msg="PLE changed a base-model weight")
    original = model.read_bytes()
    _, _, _, offsets = read_gguf(model)
    mutations = [("tidal.ple.ngram_size", 0, "I", 1),
                 ("tidal.ple.ngram_size", 0, "I", 0xFFFFFFFF),
                 ("tidal.ple.embedding_dim", 0, "I", 31),
                 ("tidal.ple.heads_per_ngram", 0, "I", 0),
                 ("tidal.ple.conv_kernel", 0, "I", 0),
                 ("tidal.ple.eos_token_id", 0, "I", 256),
                 ("tidal.ple.layers", 12, "I", 3),  # QSA, not GDN
                 ("tidal.ple.layers", 16, "I", 0),  # duplicate layer
                 ("tidal.ple.layers", 16, "I", 4),
                 ("tidal.ple.0.multipliers", 0, "I", 11),  # wrong array element type
                 ("tidal.ple.0.multipliers", 12, "Q", (1 << 64) - 1),
                 ("tidal.ple.0.vocab_sizes", 12, "Q", 0),
                 ("tidal.ple.0.offsets", 12, "Q", 1),
                 ("tidal.ple.0.vocab", 0, "I", 1)]
    for key, relative, fmt, value in mutations:
        data = bytearray(original)
        struct.pack_into("<" + fmt, data, offsets[key] + relative, value)
        bad.write_bytes(data)
        failure = run(bins / "tidal-cli", "--model", bad, "--backend", backend, ok=False)
        assert "PLE" in failure.stderr, (key, failure.stderr)
    for before, after in [(b"tidal.ple.layers", b"tidal.xle.layers"),
                          (b"tidal.ple.conv_kernel", b"tidal.ple.conv_kerneX"),
                          (b"blk.0.ple.key", b"blk.0.ple.bad")]:
        bad.write_bytes(original.replace(before, after))
        run(bins / "tidal-cli", "--model", bad, "--backend", backend, ok=False)
    for extra in [["--ple-layers", ""], ["--ple-layers", "0,"], ["--ple-layers", "0,0"],
                  ["--ple-layers", "2,0"], ["--ple-layers", "3"], ["--ple-layers", "-1"],
                  ["--ple-layers", "4"], ["--ple-dim", "128"],
                  ["--ple-layers", "0", "--ple-ngram", "2147483647"],
                  ["--ple-layers", "0", "--ple-dim", "33"],
                  ["--ple-layers", "0", "--ple-vocab-base", "0"],
                  ["--ple-layers", "0", "--ple-conv", "0"]]:
        run(bins / "tidal-dummy", "--output", root / "invalid-ple.gguf", *extra, ok=False)
    print("Malformed PLE metadata, missing tensors and generator arguments rejected", flush=True)


def check_rejections(root, bins, backend):
    cli, model = bins / "tidal-cli", root / "invalid-input.gguf"
    run(bins / "tidal-dummy", "--output", model, "--type", "f32", "--context", 32)
    common = [cli, "--model", model, "--backend", backend]
    for extra in [["--tokens", "-1"], ["--tokens", "256"], ["--tokens", "1,"],
                  ["--tokens", "1,2x"], ["--steps", "-1"], ["--steps", "32"],
                  ["--threads", "0"], ["--tokens", ""], ["--unknown", "x"]]:
        run(*common, *extra, ok=False)
    run(cli, "--model", model, "--backend", "imaginary", ok=False)
    failure = run(cli, "--model", model, "--backend", "cuda", "--device", 2147483647, ok=False)
    assert "CUDA" in failure.stderr
    bad = root / "bad.gguf"
    bad.write_bytes(model.read_bytes()[:128])
    run(cli, "--model", bad, "--backend", backend, ok=False)
    data = bytearray(model.read_bytes())
    _, _, _, offsets = read_gguf(model)
    struct.pack_into("<I", data, offsets["tidal.q_heads"], 3)
    bad.write_bytes(data)
    run(cli, "--model", bad, "--backend", backend, ok=False)
    bad.write_bytes(model.read_bytes()[:-1024])
    run(cli, "--model", bad, "--backend", backend, ok=False)
    print("CLI, malformed/truncated GGUF and strict backend rejection passed", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--formats", nargs="+", choices=FORMATS, default=FORMATS)
    args = parser.parse_args()
    failures = []
    with tempfile.TemporaryDirectory(prefix="tidal-tests-") as tmp:
        root = Path(tmp)
        cases = [(kind, lambda kind=kind: check_format(root, args.bin_dir, args.backend, kind))
                 for kind in dict.fromkeys(args.formats)]
        cases += [(f"ple-{kind}", lambda kind=kind: check_ple(root, args.bin_dir, args.backend, kind))
                  for kind in dict.fromkeys(args.formats)]
        cases += [(f"ple-{variant}", lambda variant=variant: check_ple(root, args.bin_dir, args.backend, "f32", variant))
                  for variant in ["alternate", "pointwise"]]
        cases += [(f"ple-quantized-{kind}", lambda kind=kind: check_ple(root, args.bin_dir, args.backend, kind, "quantized"))
                  for kind in dict.fromkeys(args.formats) if kind in ("q4_0", "q8_0")]
        cases += [("multi", lambda: check_multi(root, args.bin_dir, args.backend)),
                  ("wide-heads", lambda: check_wide_heads(root, args.bin_dir, args.backend)),
                  ("ple-rejections", lambda: check_ple_rejections(root, args.bin_dir, args.backend)),
                  ("rejections", lambda: check_rejections(root, args.bin_dir, args.backend))]
        for name, check in cases:
            try:
                check()
            except Exception:
                failures.append(name)
                traceback.print_exc()
    if failures:
        sys.exit("Failed cases: " + ", ".join(failures))


if __name__ == "__main__":
    main()
