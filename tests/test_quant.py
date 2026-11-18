#!/usr/bin/env python3
"""All storage formats, mixed tensors, native kernels and streamed GGUF I/O."""
import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path
from test_reference import run
from gguf_metadata import read_header


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").digest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    b = args.bin_dir
    formats = [line.split()[0] for line in run(b / "tidal-dummy", "--list-types", 1).stdout.splitlines()]
    assert len(formats) == 28 and {"q6_K", "iq1_s", "tq1_0", "nvfp4", "q1_0", "q2_0"} <= set(formats)
    with tempfile.TemporaryDirectory(prefix="tidal-quant-") as tmp:
        root = Path(tmp)
        common = [b / "tidal-dummy", "--preset", "quant-test", "--context", 32]
        base = root / "base.gguf"
        run(*common, "--output", base)
        _, tensors, _ = read_header(base)
        candidates = [n for n, t in tensors.items() if t["type"] == 0 and t["shape"][0] % 256 == 0 and ".norm" not in n]
        candidates.sort(key=lambda n: (math.prod(tensors[n]["shape"]), n))
        rules = root / "mixed.types"
        rules.write_text("# last match wins\n*.ffn.*=f16\n" + "\n".join(f"{n}={fmt}" for n, fmt in zip(candidates, formats))
                         + "\ntoken_embd.weight=nvfp4\nhead.norm=q4_0\n")
        model = root / "mixed.gguf"
        run(*common, "--output", model, "--tensor-types", rules)
        _, mixed, _ = read_header(model)
        assert len({mixed[n]["type"] for n in candidates[:len(formats)]}) == len(formats)
        models = [model]
        # All types also occupy actual [in, out, expert] tensors across three files.
        moe_base = root / "moe-base.gguf"
        run(*common, "--output", moe_base, "--experts", 4)
        _, moe, _ = read_header(moe_base)
        expert_names = [n for n in moe if n.endswith("_exps")]
        for group, first in enumerate(range(0, len(formats), len(expert_names))):
            rule = root / f"moe-{group}.types"
            rule.write_text("\n".join(f"{n}={fmt}" for n, fmt in zip(expert_names, formats[first:])))
            path = root / f"moe-{group}.gguf"
            run(*common, "--output", path, "--experts", 4, "--tensor-types", rule)
            models.append(path)
            # Expert count one still uses MUL_MAT_ID, not a Dense matvec.
            single = root / f"moe-single-{group}.gguf"
            run(*common, "--output", single, "--experts", 1, "--experts-used", 1, "--tensor-types", rule)
            models.append(single)
        for path in models:
            run(b / "tidal-weights-test", path, args.backend)
            trace = root / "trace.json"
            run(b / "tidal-cli", "--model", path, "--backend", args.backend, "--tokens", "1,7,23,9,42,5,6,2,77,13,84,11",
                "--steps", 4, "--repeat", 2, "--json", trace)
            result = json.loads(trace.read_text())
            assert all(math.isfinite(x) for step in result["trace"] for x in step["logits"])
            assert result["model"]["resident_weight_bytes"] >= result["model"]["stored_weight_bytes"]
            run(b / "tidal-state-test", path, args.backend)
            print(f"{path.stem}: all uploaded bytes, native arithmetic, inference and reset passed", flush=True)
        # Chunk/thread choices must not alter seeded weights or encoded bytes.
        a, z = root / "chunks-a.gguf", root / "chunks-z.gguf"
        for path, chunk, threads in [(a, 1, 1), (z, 16, 4)]:
            run(*common, "--hidden", 512, "--type", "q6_K", "--output", path, "--chunk-mib", chunk, "--threads", threads)
        assert digest(a) == digest(z)
        original = digest(z)
        run(*common, "--output", z, ok=False)
        assert digest(z) == original
        invalid_rules = root / "invalid.types"
        for text in ["not_a_tensor=q4_0", "head.up=unknown", "head.up", "=f32"]:
            invalid_rules.write_text(text)
            run(*common, "--output", root / "bad.gguf", "--tensor-types", invalid_rules, ok=False)
        for kind in ["q8_1", "q8_K", "i8", "q4_0_4_4"]:
            run(*common, "--type", kind, "--output", root / "bad.gguf", ok=False)
        run(b / "tidal-dummy", "--type", "q6_K", "--output", root / "bad.gguf", ok=False)  # width 128
        dry = root / "dry.gguf"
        run(*common, "--output", dry, "--dry-run", 1)
        assert not dry.exists() and not list(root.glob("*.partial"))
    print(f"{args.backend}: all 28 weight formats, mixed Dense/MoE and streaming checks passed")


if __name__ == "__main__":
    main()
