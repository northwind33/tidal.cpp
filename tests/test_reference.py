#!/usr/bin/env python3
"""Run the actual binaries and compare full trajectories to an independent oracle."""
import argparse
import json
import os
import struct
import subprocess
import tempfile
from pathlib import Path

# Tiny matrices are much faster with one BLAS thread, independently of ggml threads.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from reference import Reference, read_gguf


def run(*args, ok=True):
    result = subprocess.run([str(a) for a in args], capture_output=True, text=True, timeout=120)
    if (result.returncode == 0) != ok:
        raise AssertionError(f"{' '.join(map(str, args))}\n{result.stdout}\n{result.stderr}")
    # A rejected CLI input should be a catchable error, never a ggml assertion/abort.
    if not ok:
        assert result.returncode == 1, result.stderr
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    cli, dummy = args.bin_dir / "tidal-cli", args.bin_dir / "tidal-dummy"
    with tempfile.TemporaryDirectory(prefix="tidal-tests-") as tmp:
        root = Path(tmp)
        prompt = [1, 7, 23, 9, 42, 5, 6, 2, 77, 13, 84, 11, 90, 3, 42, 28, 61, 7, 0, 93, 34, 22, 1, 255]
        for kind in ["f32", "f16", "bf16", "q8_0", "q4_0"]:
            model, trace = root / f"{kind}.gguf", root / f"{kind}.json"
            run(dummy, "--output", model, "--type", kind, "--context", 32)
            run(cli, "--model", model, "--backend", args.backend, "--tokens", ",".join(map(str, prompt)),
                "--steps", 6, "--json", trace, "--repeat", 2, "--threads", 2)
            result = json.loads(trace.read_text())
            assert result["sparse_attention_backend"] == "CPU"
            assert result["kv_type"] == "BF16"
            assert result["statistics"]["sparse_attention_calls"] == 30
            if args.backend == "cuda":
                assert result["backend"].startswith("CUDA")
                assert result["statistics"]["device_to_host_bytes"] > 0
                assert result["statistics"]["host_to_device_bytes"] > 0
            reference = Reference(model)
            max_error = 0
            for i, token in enumerate(prompt + result["generated"]):
                expected, selected = reference.decode(token)
                actual = np.array(result["trace"][i]["logits"])
                max_error = max(max_error, float(np.max(np.abs(expected - actual))))
                # Quantized matvecs also round activations. One near-threshold
                # rounding difference can propagate through later recurrent steps.
                # F32 checks the equations tightly; low-precision checks retain
                # an absolute bound plus a whole-vocabulary normalized RMS bound.
                tolerance = {"f32": 2e-4, "f16": 3e-3, "bf16": 1.2e-2,
                             "q8_0": 2.5e-2, "q4_0": 2.5e-2}[kind]
                if args.backend == "cuda":
                    tolerance = 3e-2  # backend-specific activation quantization; still checks full logits
                np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance,
                    err_msg=f"{kind}, token position {i}")
                nrmse = np.sqrt(np.mean((actual - expected) ** 2) / np.mean(expected ** 2))
                assert nrmse < 0.025, (kind, i, nrmse)
                assert result["trace"][i]["position"] == i
                assert result["trace"][i]["selections"] == selected, (kind, i, result["trace"][i]["selections"], selected)
            print(f"{kind}: 30-token logits + QSA selections + reset/replay passed; max_abs={max_error:.8g}")

        # A second superblock catches per-layer cache aliasing.
        multi, trace = root / "multi.gguf", root / "multi.json"
        run(dummy, "--output", multi, "--layers", 8, "--context", 32, "--seed", 123, "--hidden", 256)
        run(cli, "--model", multi, "--backend", args.backend, "--tokens", ",".join(map(str, prompt)),
            "--steps", 0, "--json", trace)
        result, reference = json.loads(trace.read_text()), Reference(multi)
        for i, token in enumerate(prompt):
            expected, selected = reference.decode(token)
            np.testing.assert_allclose(result["trace"][i]["logits"], expected, atol=3e-4, rtol=3e-4)
            assert result["trace"][i]["selections"] == selected
        print("8 layers / hidden 256: two independent QSA caches passed")

        model = root / "f32.gguf"
        common = [cli, "--model", model, "--backend", args.backend]
        for extra in [["--tokens", "-1"], ["--tokens", "256"], ["--tokens", "1,"],
                      ["--tokens", "1,2x"], ["--steps", "-1"], ["--steps", "32"],
                      ["--threads", "0"], ["--tokens", ""], ["--unknown", "x"]]:
            run(*common, *extra, ok=False)
        run(cli, "--model", model, "--backend", "imaginary", ok=False)
        if args.backend == "cpu":
            failure = run(cli, "--model", model, "--backend", "cuda", "--device", 2147483647, ok=False)
            assert "CUDA" in failure.stderr
        bad = root / "bad.gguf"
        bad.write_bytes(model.read_bytes()[:128])
        run(cli, "--model", bad, "--backend", args.backend, ok=False)
        data = bytearray(model.read_bytes())
        _, _, _, offsets = read_gguf(model)
        struct.pack_into("<I", data, offsets["tidal.q_heads"], 3)
        bad.write_bytes(data)
        run(cli, "--model", bad, "--backend", args.backend, ok=False)
        bad.write_bytes(model.read_bytes()[:-1024])
        run(cli, "--model", bad, "--backend", args.backend, ok=False)
        print("CLI, malformed/truncated GGUF and strict backend rejection passed")


if __name__ == "__main__":
    main()
