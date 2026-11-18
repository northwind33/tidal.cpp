#!/usr/bin/env python3
"""Opt-in 27B integration validation; all artifacts stay in the output directory."""
import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
from gguf_metadata import read_header
from test_reference import check_placement


def gpu_processes():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True,
                            check=True, timeout=10)
    memory = {}
    for line in result.stdout.splitlines():
        fields = [s.strip() for s in line.split(',')]
        if len(fields) == 2:
            pid, mib = map(int, fields)
            memory[pid] = memory.get(pid, 0) + mib
    return memory


def monitored(command, log_path, monitor_gpu=False):
    peak_mib = 0
    baseline = set(gpu_processes()) if monitor_gpu else set()
    observed_pid = None
    start = time.monotonic()
    with log_path.open("w") as log:
        process = subprocess.Popen(list(map(str, command)), stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                if time.monotonic() - start > 1800:
                    raise TimeoutError("Large-model validation exceeded 30 minutes")
                if monitor_gpu:
                    memory = gpu_processes()
                    new_pids = set(memory) - baseline
                    if observed_pid is None and new_pids:
                        if process.pid in new_pids:
                            observed_pid = process.pid
                        elif len(new_pids) == 1:
                            # NVML reports host PIDs inside a PID namespace.
                            # Associate the sole newly created GPU process; no
                            # concurrent GPU jobs may start during this test.
                            observed_pid = next(iter(new_pids))
                        else:
                            raise RuntimeError("Ambiguous GPU process IDs; rerun without concurrent GPU jobs")
                    if observed_pid != process.pid and new_pids - {observed_pid}:
                        raise RuntimeError("Concurrent GPU process prevents namespace-safe memory attribution")
                    peak_mib = max(peak_mib, memory.get(observed_pid, 0))
                time.sleep(.5)
            if process.returncode:
                raise RuntimeError(f"Validation process exited {process.returncode}; see {log_path}")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    return dict(wall_seconds=time.monotonic() - start, sampled_peak_vram_mib=peak_mib)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("build/large-validation"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata, tensors, data_offset = read_header(args.model)
    config = {key[6:]: value for key, value in metadata.items() if key.startswith("tidal.")}
    parameters = sum(math.prod(t["shape"]) for t in tensors.values())
    assert 26.8e9 <= parameters <= 27.3e9, parameters
    assert metadata["general.architecture"] == "tidal-qwen4exp-dense"
    assert metadata["tidal.residual_init_scale"] == 1 / metadata["tidal.layers"]
    assert not any(key.startswith("tidal.ple.") or key == "tidal.experts" for key in metadata)
    assert len({t["type"] for t in tensors.values()}) >= 3
    bpw = 8 * (args.model.stat().st_size - data_offset) / parameters
    assert 5.8 <= bpw <= 6.4, bpw
    prompt = [1, 7, 23, 9, 42, 5, 6, 2, 77, 13, 84, 11, 90, 3, 42, 28]
    gpu_trace, cpu_trace = args.output_dir / "cuda.trace.json", args.output_dir / "cpu.trace.json"
    cli = args.bin_dir / "tidal-cli"
    gpu_time = monitored([cli, "--model", args.model, "--backend", "cuda", "--tokens", ",".join(map(str, prompt)),
                          "--steps", 2, "--repeat", 2, "--json", gpu_trace], args.output_dir / "cuda.log", True)
    gpu = json.loads(gpu_trace.read_text())
    assert gpu["backend"].startswith("CUDA")
    assert gpu["model"]["parameters"] == parameters and gpu["model"]["converted_tensors"] == 0
    assert gpu["model"]["resident_weight_bytes"] == gpu["model"]["stored_weight_bytes"]
    assert 0 < gpu_time["sampled_peak_vram_mib"] < 24 * 1024
    assert gpu_time["sampled_peak_vram_mib"] * 1024**2 >= gpu["model"]["resident_weight_bytes"]
    sequence = prompt + gpu["generated"]
    check_placement(gpu, config, "cuda", len(sequence))
    # Teacher-force identical tokens: CPU/CUDA native quantizers need not choose
    # the same greedy token when large-vocabulary logits are nearly tied.
    cpu_time = monitored([cli, "--model", args.model, "--backend", "cpu", "--tokens", ",".join(map(str, sequence)),
                          "--steps", 0, "--threads", 4, "--json", cpu_trace], args.output_dir / "cpu.log")
    cpu = json.loads(cpu_trace.read_text())
    assert cpu["backend"].startswith("CPU") and len(cpu["trace"]) == len(gpu["trace"]) == len(sequence)
    assert cpu["model"]["converted_tensors"] == 0
    check_placement(cpu, config, "cpu", len(sequence))
    max_nrmse, max_abs, min_cosine = 0., 0., 1.
    for reference, actual in zip(cpu["trace"], gpu["trace"]):
        assert reference["position"] == actual["position"]
        assert reference["selections"] == actual["selections"]
        a, z = np.asarray(reference["logits"]), np.asarray(actual["logits"])
        assert np.isfinite(a).all() and np.isfinite(z).all()
        nrmse = float(np.linalg.norm(a - z) / np.linalg.norm(a))
        cosine = float(np.dot(a, z) / (np.linalg.norm(a) * np.linalg.norm(z)))
        assert nrmse < .05 and cosine > .995, (actual["position"], nrmse, cosine)
        max_nrmse = max(max_nrmse, nrmse)
        max_abs = max(max_abs, float(np.max(np.abs(a - z))))
        min_cosine = min(min_cosine, cosine)
    report = dict(parameters=parameters, file_bytes=args.model.stat().st_size, effective_bpw=bpw,
                  synthetic_seed=config["seed"], residual_init_scale=config["residual_init_scale"],
                  weight_info=gpu["model"], cuda=gpu_time, cpu=cpu_time, compared_tokens=len(sequence),
                  generated=gpu["generated"], max_nrmse=max_nrmse, max_abs=max_abs, min_cosine=min_cosine,
                  reset_replay=True, qsa_selections_equal=True)
    path = args.output_dir / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Large-model validation passed; local report: {path}")


if __name__ == "__main__":
    main()
