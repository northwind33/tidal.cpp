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
    for name in ["blk.0.hc_attn.down", "blk.0.hc_attn.inject", "blk.0.hc_attn.up",
                 "blk.0.gdn.qkv", "output.weight"]:
        width = reference.w[name].shape[1]
        mixed = rng.uniform(-2, 2, width).astype(np.float32)
        if width > 32:
            mixed[:32] = 0  # Zero activation block must not divide by zero.
        below_half = np.nextafter(np.float32(0.5), np.float32(0))
        ties = np.resize(np.array([127, -127, 0.5, -0.5, 1.5, -1.5, below_half, -below_half],
                                  dtype=np.float32), width)
        for x in [np.zeros(width, dtype=np.float32), mixed, ties]:
            requests.append(name + " " + " ".join(map(str, x)))
            expected.append((name, reference.mm(name, x)))
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
    assert stats["device_to_host_bytes"] == (d2h if backend == "cuda" else 0), stats
    assert stats["host_to_device_bytes"] == (h2d if backend == "cuda" else 0), stats


def check_trace(result, reference, tolerance, label):
    sequence = PROMPT + result["generated"]
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
        if len(PROMPT) - 1 <= i < len(sequence) - 1:
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
        cases += [("multi", lambda: check_multi(root, args.bin_dir, args.backend)),
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
