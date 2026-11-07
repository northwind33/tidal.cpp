#!/usr/bin/env python3
"""Compare compatible tidal-bench reports; optionally fail on throughput regression."""
import argparse
import json
import math
import sys
from pathlib import Path


def compare(before, after, allow_sampler_change=False):
    for value in (before, after):
        if value.get("schema_version") != 1 or value.get("kind") != "tidal-benchmark":
            raise ValueError("Expected tidal-benchmark schema_version 1")
    for section, keys in {
        "model": ["fnv1a64", "size_bytes", "config"],
        "environment": ["backend", "device", "device_index", "cpu", "threads", "nvtx",
                        "cuda_driver_version", "cuda_runtime_version"],
        "workload": ["prompt", "steps", "warmup"],
    }.items():
        for key in keys:
            if before[section][key] != after[section][key]:
                raise ValueError(f"Incompatible reports: {section}.{key} differs")
    if before["workload"].get("capture_logits", True) != after["workload"].get("capture_logits", True):
        raise ValueError("Incompatible reports: logit capture differs")
    def prefill_settings(report):
        value = report['workload'].get('prefill')
        if value is None:
            return None
        value = dict(value)
        # Reports predating the GPU KV option always used CPU BF16 decode KV.
        value.setdefault('kv_placement', 'cpu')
        value.setdefault('kv_type', 'BF16')
        value.setdefault('decode_attention_backend', 'CPU')
        return value
    if prefill_settings(before) != prefill_settings(after):
        raise ValueError("Incompatible KV placement/type, prefill execution, precision, chunk, tile or memory settings")
    a, b = dict(before["workload"].get("sampling", {})), dict(after["workload"].get("sampling", {}))
    if allow_sampler_change:
        a.pop("backend", None); b.pop("backend", None)
    if a != b:
        raise ValueError("Incompatible sampling settings; use --allow-sampler-change only to compare CPU/device execution")
    if not before["runs"] or not after["runs"]:
        raise ValueError("Empty measured runs")
    expected = before["runs"][0]["generated"]
    for value in (before, after):
        if any(run["generated"] != expected for run in value["runs"]):
            raise ValueError("Generated tokens differ; the decode workloads are not comparable")
    rows = []
    for phase in ("prefill", "decode", "total"):
        a, b = before["summary"][phase], after["summary"][phase]
        old, new = a["tokens_per_second"], b["tokens_per_second"]
        if old is None and new is None:
            continue
        if not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in (old, new)):
            raise ValueError(f"Invalid throughput for {phase}")
        rows.append({"phase": phase, "before_tokens_per_second": old, "after_tokens_per_second": new,
                     "throughput_change_percent": 100 * (new / old - 1),
                     "before_p95_ms": (a["token_latency_ms"] or {}).get("p95"),
                     "after_p95_ms": (b["token_latency_ms"] or {}).get("p95")})
    hotspots = []
    p, q = before.get("profile"), after.get("profile")
    if p and q and p["mode"] == q["mode"] and not p["dropped_events"] and not q["dropped_events"]:
        def key(row):
            return tuple(row[k] for k in ("category", "phase", "layer", "name", "op"))
        previous = {key(row): row for row in p["aggregates"]}
        for row in q["aggregates"]:
            old = previous.get(key(row))
            if old and old["wall_us"]["count"] == row["wall_us"]["count"]:
                hotspots.append({"category": row["category"], "phase": row["phase"], "layer": row["layer"],
                                 "name": row["name"], "op": row["op"],
                                 "before_us": old["total_us"], "after_us": row["total_us"],
                                 "delta_us": row["total_us"] - old["total_us"]})
        hotspots.sort(key=lambda row: abs(row["delta_us"]), reverse=True)
    return {"schema_version": 1, "metrics": rows, "profile_changes": hotspots,
            "output_checksums_match": before["runs"][0]["output_checksum"] == after["runs"][0]["output_checksum"],
            "note": "Profile totals overlap and come from separate diagnostic replays; they are not throughput timings."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--max-regression", type=float, help="Fail if any measured throughput decreases by more than this percent")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--allow-sampler-change", action="store_true", help="Allow CPU/device sampler differences; all sampling parameters and generated tokens must still match")
    args = parser.parse_args()
    if args.max_regression is not None and (not math.isfinite(args.max_regression) or args.max_regression < 0):
        parser.error("--max-regression must be finite and nonnegative")
    if args.json and args.json.resolve() in (args.baseline.resolve(), args.candidate.resolve()):
        parser.error("Comparison output must differ from both input reports")
    try:
        result = compare(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()), args.allow_sampler_change)
        for row in result["metrics"]:
            latency = (f"; token p95 {row['before_p95_ms']:.3f} -> {row['after_p95_ms']:.3f} ms"
                       if row['before_p95_ms'] is not None and row['after_p95_ms'] is not None else "")
            print(f"{row['phase']}: {row['before_tokens_per_second']:.3f} -> {row['after_tokens_per_second']:.3f} tokens/s "
                  f"({row['throughput_change_percent']:+.2f}%)" + latency)
        if not result["output_checksums_match"]:
            print("Output checksums differ; run numerical validation to assess accuracy.")
        for row in result["profile_changes"][:15]:
            print(f"profile {row['delta_us']:+.3f} us: layer={row['layer']} {row['phase']}/{row['category']}/{row['name']}")
        if args.json:
            args.json.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        if args.max_regression is not None and any(row["throughput_change_percent"] < -args.max_regression for row in result["metrics"]):
            return 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"compare_bench: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
