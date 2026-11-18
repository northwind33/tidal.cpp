#!/usr/bin/env python3
"""Check the 27B recipe and overflow bounds without allocating model payloads."""
import argparse
import re
import subprocess
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin-dir", type=Path, required=True)
    args = parser.parse_args()
    recipe = Path(__file__).resolve().parents[1] / "configs" / "dense-27b-6bpw.types"
    with tempfile.TemporaryDirectory(prefix="tidal-plan-") as directory:
        target = Path(directory) / "large.gguf"
        base = [str(args.bin_dir / "tidal-dummy"), "--preset", "dense-27b", "--type", "q6_K",
                "--tensor-types", str(recipe), "--dry-run", "1", "--output", str(target)]
        result = subprocess.run(base, text=True, capture_output=True, check=True, timeout=30)
        match = re.search(r"parameters=(\d+) tensor_bytes=(\d+) effective_bpw=([\d.]+)", result.stdout)
        assert match and int(match[1]) == 27108415232 and int(match[2]) == 20763849728
        assert 6.12 < float(match[3]) < 6.14 and not target.exists()
        for option, value in [("--hidden", "2147483647"), ("--layers", "129"), ("--vocab", "262145"),
                              ("--ffn", "131073"), ("--rank", "2147483647")]:
            invalid = subprocess.run(base + [option, value], text=True, capture_output=True, timeout=30)
            assert invalid.returncode == 1 and not target.exists(), invalid.stderr
    print("27B mixed-format plan and dimension bounds passed without payload allocation")


if __name__ == "__main__":
    main()
