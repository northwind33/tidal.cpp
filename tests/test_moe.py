#!/usr/bin/env python3
"""Independent router, sparse experts, shared expert and state regressions."""
import argparse
import json
import struct
import tempfile
from pathlib import Path
from test_reference import run, check_trace, check_placement, Reference, read_gguf, PROMPT, TOLERANCES, np


def check_experts(binary, model, backend):
    ref = Reference(model, backend)
    rng = np.random.default_rng(42)
    requests, expected = [], []
    for part in ["gate_exps", "up_exps", "down_exps"]:
        name = "blk.0.ffn." + part
        for expert in range(ref.c["experts"]):
            x = rng.uniform(-1, 1, ref.w[name].shape[-1]).astype(np.float32)
            requests.append(f"@expert {name} {expert} " + " ".join(map(str, x)))
            expected.append(ref.mm(name, x, expert))
    result = run(binary, model, backend, input="\n".join(requests) + "\n")
    lines = result.stdout.splitlines()
    assert len(lines) == len(expected)
    for line, values in zip(lines, expected):
        actual = np.array(line.split()[1:], dtype=np.float32)
        # Native expert kernels use default accumulation, unlike Dense matvecs.
        tolerance = 2e-3 if backend == "cuda" and ref.types["blk.0.ffn.gate_exps"] == 1 else 2e-5
        np.testing.assert_allclose(actual, values, atol=tolerance, rtol=tolerance)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    b = args.bin_dir
    with tempfile.TemporaryDirectory(prefix="tidal-moe-") as tmp:
        root = Path(tmp)
        cases = [("top2", "f32", []), ("top1", "f32", ["--experts-used", 1]),
                 ("all", "f32", ["--experts-used", 4]),
                 ("unnormalized", "f32", ["--norm-topk", 0]),
                 ("no-shared", "f32", ["--shared-ffn", 0]),
                 ("ple", "f32", ["--ple-layers", "0,2"]),
                 ("f16", "f16", []), ("bf16", "bf16", []),
                 ("q4", "q4_0", []), ("q8", "q8_0", [])]
        for label, kind, extra in cases:
            model, trace = root / f"{label}.gguf", root / f"{label}.json"
            run(b / "tidal-dummy", "--output", model, "--experts", 4, "--context", 32, "--type", kind, *extra)
            run(b / "tidal-cli", "--model", model, "--backend", args.backend,
                "--tokens", ",".join(map(str, PROMPT)), "--steps", 6, "--repeat", 2, "--json", trace)
            result, ref = json.loads(trace.read_text()), Reference(model, args.backend)
            check_experts(b / "tidal-matvec-test", model, args.backend)
            if kind == "f32":
                check_trace(result, ref, TOLERANCES[args.backend][kind], "MoE " + label)
            else:
                assert all(np.isfinite(step["logits"]).all() for step in result["trace"])
            check_placement(result, ref.c, args.backend, 30)
            run(b / "tidal-state-test", model, args.backend)
            print(f"MoE {label}: {'trajectory reference' if kind == 'f32' else 'native expert reference'}, generation, state and placement passed", flush=True)
        model = root / "top2.gguf"
        _, _, _, offsets = read_gguf(model)
        for key, value in [("experts", 0), ("experts_used", 0), ("experts_used", 5),
                           ("expert_ffn", 33), ("shared_ffn", 0xFFFFFFFF)]:
            data = bytearray(model.read_bytes())
            struct.pack_into("<I", data, offsets["tidal." + key], value)
            invalid = root / "invalid.gguf"
            invalid.write_bytes(data)
            run(b / "tidal-cli", "--model", invalid, "--backend", args.backend, ok=False)
        for extra in [["--experts", 1], ["--experts", -1], ["--experts-used", 1],
                      ["--experts", 4, "--norm-topk", 2], ["--experts", 4, "--expert-ffn", 0]]:
            run(b / "tidal-dummy", "--output", root / "bad.gguf", *extra, ok=False)
        profile_path = root / "profile.json"
        run(b / "tidal-bench", "--model", model, "--backend", args.backend, "--prompt", 3, "--steps", 1,
            "--warmup", 0, "--repetitions", 1, "--profile", "nodes", "--json", profile_path)
        profile = json.loads(profile_path.read_text())["profile"]
        nodes = [e for e in profile["events"] if e["cat"] == "node"]
        assert any(e["args"]["op"] == "MUL_MAT_ID" for e in nodes)
        assert profile["logits_max_abs_vs_baseline"] < 5e-4
        assert all(e["args"]["backend"].startswith("CUDA" if args.backend == "cuda" else "CPU") for e in nodes)
    print("MoE metadata rejection and profiler checks passed")


if __name__ == "__main__":
    main()
