#!/usr/bin/env python3
"""Exercise benchmark accounting, profiling isolation and baseline comparison."""
import argparse
import copy
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path


def run(*args, code=0):
    p = subprocess.run(list(map(str, args)), capture_output=True, text=True, timeout=120)
    assert p.returncode == code, (args, p.returncode, p.stdout, p.stderr)
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    bench, dummy = args.bin_dir / "tidal-bench", args.bin_dir / "tidal-dummy"
    compare = Path(__file__).resolve().parents[1] / "tools" / "compare_bench.py"
    with tempfile.TemporaryDirectory(prefix="tidal-bench-tests-") as directory:
        root = Path(directory)
        model = root / 'model-"quoted.gguf'
        run(dummy, "--output", model, "--context", 32)
        base = [bench, "--model", model, "--backend", args.backend, "--prompt", 8, "--steps", 4,
                "--warmup", 1, "--repetitions", 2, "--threads", 2]
        reports = {}
        for mode in ["off", "phases", "nodes"]:
            path, trace = root / f"{mode}.json", root / f"{mode}-trace.json"
            extra = [] if mode == "off" else ["--trace", trace]
            run(*base, "--profile", mode, "--json", path, "--label", 'test "json"\nlabel', *extra)
            r = json.loads(path.read_text())
            reports[mode] = r
            assert r["schema_version"] == 1 and r["workload"]["repetitions"] == 2
            assert r["label"] == 'test "json"\nlabel' and r["model"]["path"] == str(model)
            assert len(r["runs"]) == 2
            for item in r["runs"]:
                assert len(item["prefill_latency_ms"]) == 8 and len(item["decode_latency_ms"]) == 4
                assert item["statistics"]["sparse_attention_calls"] == 12
                assert len(item["generated"]) == 4
            for phase, tokens in [("prefill", 8), ("decode", 4), ("total", 12)]:
                times = [x["prefill_ms"] if phase == "prefill" else x["decode_ms"] if phase == "decode"
                         else x["prefill_ms"] + x["decode_ms"] for x in r["runs"]]
                summary = r["summary"][phase]
                assert math.isclose(summary["tokens_per_second"], 1000 * tokens * 2 / sum(times), rel_tol=1e-9)
                assert summary["token_latency_ms"]["count"] == tokens * 2
            if mode == "off":
                assert r["profile"] is None
                continue
            p = r["profile"]
            assert p["separate_replay"] and p["fixed_generated_tokens"]
            assert p["dropped_events"] == 0
            assert p["qsa_matches_baseline"] and p["greedy_matches_baseline"]
            assert p["logits_max_abs_vs_baseline"] < 5e-4
            assert p["statistics"]["sparse_attention_calls"] == 12
            events = p["events"]
            assert len([e for e in events if e["cat"] == "token"]) == 12  # No warmup/repetition leakage.
            assert {e["args"]["position"] for e in events} == set(range(12))
            names = {e["name"] for e in events}
            assert {"qk", "softmax", "weighted_v", "kv_append_bf16", "worker_launch", "worker_join"} <= names
            for event in events:
                assert math.isfinite(event["dur"]) and event["dur"] >= 0 and event["ts"] >= 0
            for name in ["qk", "softmax", "weighted_v"]:
                assert len([e for e in events if e["cat"] == "cpu_head" and e["name"] == name]) == 12 * 4
            nodes = [e for e in events if e["cat"] == "node"]
            if mode == "nodes":
                assert len(nodes) == p["statistics"]["graph_nodes"]
                assert all(len(e["args"]["shape"]) == 4 and e["args"]["op"] for e in nodes)
                assert any(e["args"]["op"] == "MUL_MAT" and e["name"] == "output.weight" for e in nodes)
            else:
                assert not nodes and p["logits_max_abs_vs_baseline"] == 0
            if args.backend == "cuda":
                for name, field in [("D2H", "d2h_bytes"), ("H2D", "h2d_bytes")]:
                    assert sum(e["args"]["bytes"] for e in events if e["cat"] == "transfer" and e["name"] == name) == p["statistics"][field]
            timeline = json.loads(trace.read_text())["traceEvents"]
            assert len(timeline) == len(events) and all(e["ph"] == "X" for e in timeline)
        checksum = reports["off"]["runs"][0]["output_checksum"]
        assert all(r["runs"][0]["output_checksum"] == checksum for r in reports.values())
        capped = root / "capped.json"
        run(*base, "--profile", "nodes", "--profile-limit", 10, "--json", capped)
        capped_profile = json.loads(capped.read_text())["profile"]
        assert len(capped_profile["events"]) == 10 and capped_profile["dropped_events"] > 0
        assert capped_profile["qsa_matches_baseline"]
        if args.backend == "cuda":
            run(bench, "--model", model, "--backend", "cuda", "--prompt", 2, "--steps", 1,
                "--warmup", 0, "--repetitions", 1, "--nvtx", 1)
        for kind in ["f16", "bf16", "q8_0", "q4_0"]:
            quantized, output = root / f"{kind}.gguf", root / f"{kind}.json"
            run(dummy, "--output", quantized, "--type", kind, "--context", 32)
            run(bench, "--model", quantized, "--backend", args.backend, "--prompt", 8, "--steps", 4,
                "--warmup", 0, "--repetitions", 1, "--profile", "nodes", "--json", output)
            profile = json.loads(output.read_text())["profile"]
            assert profile["dropped_events"] == 0 and profile["qsa_matches_baseline"]
            assert profile["logits_max_nrmse_vs_baseline"] < .025, (kind, profile["logits_max_nrmse_vs_baseline"])
        # PLE has its own graph and host hashing phase; diagnostic replay must
        # preserve the new histories, and config reports must identify the model.
        ple, output = root / "ple.gguf", root / "ple.json"
        run(dummy, "--output", ple, "--ple-layers", "0,2", "--context", 32)
        run(bench, "--model", ple, "--backend", args.backend, "--prompt", 8, "--steps", 4,
            "--warmup", 0, "--repetitions", 1, "--profile", "nodes", "--json", output)
        report = json.loads(output.read_text())
        assert [l["layer"] for l in report["model"]["config"]["ple"]["layers"]] == [0, 2]
        profile = report["profile"]
        assert profile["qsa_matches_baseline"] and profile["greedy_matches_baseline"]
        assert profile["logits_max_abs_vs_baseline"] < 5e-4 and profile["dropped_events"] == 0
        events = [e for e in profile["events"] if e["args"]["phase"] == "ple"]
        assert len([e for e in events if e["name"] == "ngram_hash"]) == 24
        for layer in [0, 2]:
            nodes = [e for e in events if e["cat"] == "node" and e["args"]["layer"] == layer]
            assert all(e["args"]["backend"].startswith("CUDA" if args.backend == "cuda" else "CPU") for e in nodes)
            names = {e["name"] for e in nodes}
            assert {f"blk.{layer}.ple.{n}" for n in ["lookup", "key", "value", "gate", "conv.tap.0", "conv_silu"]} <= names
        # Validate performance regression detection deterministically, independent of wall-clock noise.
        baseline = root / "off.json"
        run(sys.executable, compare, baseline, baseline, "--max-regression", 0)
        candidate = copy.deepcopy(reports["off"])
        for phase in ["prefill", "decode", "total"]:
            candidate["summary"][phase]["tokens_per_second"] *= 0.8
        changed = root / "changed.json"
        changed.write_text(json.dumps(candidate))
        run(sys.executable, compare, baseline, changed, "--max-regression", 10, code=1)
        candidate["environment"]["threads"] += 1
        changed.write_text(json.dumps(candidate))
        run(sys.executable, compare, baseline, changed, code=2)
        # Zero generated tokens and bad requests have explicit behavior.
        zero = root / "zero.json"
        run(bench, "--model", model, "--backend", args.backend, "--tokens", "1,2", "--steps", 0,
            "--warmup", 0, "--repetitions", 1, "--json", zero)
        z = json.loads(zero.read_text())
        assert z["summary"]["decode"]["tokens_per_second"] is None
        assert z["summary"]["decode"]["token_latency_ms"] is None
        for extra in [["--profile", "invalid"], ["--trace", root / "bad.json"], ["--json", model],
                      ["--tokens", "1,"], ["--nvtx", 2]]:
            run(*base, *extra, code=1)
        run(bench, "--model", model, "--backend", args.backend, "--repetitions", 0, code=1)
        run(bench, "--model", model, "--backend", args.backend, "--prompt", 32, "--steps", 1, code=1)
    print(f"{args.backend}: benchmark accounting, phases/nodes replay, trace, comparison and invalid-input checks passed")


if __name__ == "__main__":
    main()
