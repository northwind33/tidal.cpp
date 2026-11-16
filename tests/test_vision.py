#!/usr/bin/env python3
"""Vision input: image decoding, the vision tower against a NumPy oracle,
prompts with images through every prefill path against the NumPy text oracle
(image rows, interleaved MRoPE, QSA block keys at image positions), the prompt
cache and the HTTP API, on a synthetic model and a synthetic vision tower.
"""
import argparse
import base64
import io
import json
import re
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

import image_codecs as codecs
from reference import Reference
from test_server import Server
from vision_reference import VisionReference

IMAGE_TOKEN = 200
# Prompt: text, image A (3x2 tokens), text, image B (2x4 tokens), text. With
# QSA blocks of 4 tokens, blocks start inside images and span their borders.
PROMPT = [5, 17, 99, "A", 33, 4, "B", 61, 2, 9]
STEPS = 4


def run(*args, ok=True):
    result = subprocess.run([str(a) for a in args], capture_output=True, text=True, timeout=300)
    if ok and result.returncode:
        raise AssertionError(f"{args[0]} failed ({result.returncode}):\n{result.stdout}\n{result.stderr}")
    if not ok and not result.returncode:
        raise AssertionError(f"{args[0]} unexpectedly succeeded:\n{result.stdout}")
    return result


def test_image(width, height, seed):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]
    image = np.stack([128 + 90 * np.sin(x / (3.0 + seed)), 128 + 90 * np.cos(y / 5.0), (x * 7 + y * 3 + 40 * seed) % 256], -1)
    return np.clip(image + rng.normal(0, 20, image.shape), 0, 255).astype(np.uint8)


def decode(bins, root, paths):
    prefix = root / "decoded"
    result = json.loads(run(bins / "tidal-vision", "--images", ",".join(map(str, paths)), "--dump-resized", prefix).stdout)
    out = []
    for i, info in enumerate(result["images"]):
        if "error" in info:
            out.append(info["error"])
            continue
        data = np.fromfile(f"{prefix}.{i}.input.rgb", dtype=np.uint8)
        out.append(data.reshape(info["height"], info["width"], 3))
    return out


def check_decoding(root, bins):
    rng = np.random.default_rng(7)
    h, w = 23, 37
    rgb = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    alpha = rng.integers(0, 256, (h, w), dtype=np.uint8)
    gray16 = rng.integers(0, 65536, (h, w), dtype=np.uint16)
    rgb16 = rng.integers(0, 65536, (h, w, 3), dtype=np.uint16)
    palette = rng.integers(0, 256, (16, 3), dtype=np.uint8)
    index = rng.integers(0, 16, (h, w), dtype=np.uint8)
    trns = rng.integers(0, 256, 16, dtype=np.uint8)
    cases = []  # (path, expected RGB)

    def png(name, samples, expected, color, depth=8, **kw):
        for interlace in (False, True):
            path = root / f"{name}{'-adam7' if interlace else ''}.png"
            codecs.write_png(path, samples, color, depth, interlace=interlace, **kw)
            cases.append((path, expected))

    png("rgb", rgb, rgb, 2)
    png("rgba", np.dstack([rgb, alpha]), codecs.over_white(rgb, alpha), 6)
    png("rgb16", rgb16, (rgb16 >> 8).astype(np.uint8), 2, 16)
    png("rgba16", np.dstack([rgb16, gray16]), codecs.over_white((rgb16 >> 8).astype(np.uint8), (gray16 >> 8).astype(np.uint8)), 6, 16)
    for depth in (1, 2, 4, 8):
        g = rng.integers(0, 1 << depth, (h, w), dtype=np.uint16)
        value = (g * 255 // ((1 << depth) - 1)).astype(np.uint8)
        png(f"gray{depth}", g, np.dstack([value] * 3), 0, depth)
    png("gray16", gray16, np.dstack([(gray16 >> 8).astype(np.uint8)] * 3), 0, 16)
    gray8 = (gray16 >> 8).astype(np.uint8)
    png("graya", np.dstack([gray8, alpha]), codecs.over_white(np.dstack([gray8] * 3), alpha), 4)
    png("palette", index, codecs.over_white(palette[index], trns[index]), 3, 8, palette=palette, trns=trns)
    png("palette4", index, palette[index], 3, 4, palette=palette)
    # Gray tRNS: one sample value is transparent.
    g = rng.integers(0, 4, (h, w), dtype=np.uint16) * 85
    value = np.where(g == 170, 255, g).astype(np.uint8)
    png("graytrns", g, np.dstack([value] * 3), 0, 8, trns=bytes([0, 170]))
    codecs.write_bmp(root / "rgb.bmp", rgb)
    cases.append((root / "rgb.bmp", rgb))
    codecs.write_ppm(root / "rgb.ppm", rgb)
    cases.append((root / "rgb.ppm", rgb))
    codecs.write_ppm(root / "gray.pgm", gray16 >> 6, 1023)
    cases.append((root / "gray.pgm", np.dstack([((gray16.astype(np.int64) >> 6) * 255 + 511) // 1023] * 3).astype(np.uint8)))
    images = decode(bins, root, [path for path, _ in cases])
    for (path, expected), actual in zip(cases, images):
        assert not isinstance(actual, str), (path.name, actual)
        assert actual.shape == expected.shape and np.array_equal(actual, expected), (path.name, np.abs(actual.astype(int) - expected).max())

    # JPEG: the progressive and restart variants carry the same coefficients
    # as baseline, so they must decode identically; all stay near the source.
    y, x = np.mgrid[0:77, 0:101]
    smooth = np.clip(np.stack([128 + 100 * np.sin(x / 9.0), 128 + 100 * np.cos(y / 7.0), (x * 2 + y * 3) % 256], -1), 0, 255)
    variants = {"base444": dict(subsample=False), "base420": dict(), "base420r": dict(restart=3), "prog420": dict(progressive=True),
                "prog444r": dict(progressive=True, subsample=False, restart=5), "gray": dict(gray=True),
                "grayprog": dict(gray=True, progressive=True)}
    for name, options in variants.items():
        codecs.write_jpeg(root / f"{name}.jpg", smooth, quality=92, **options)
    decoded = dict(zip(variants, decode(bins, root, [root / f"{name}.jpg" for name in variants])))
    assert np.array_equal(decoded["base420"], decoded["base420r"]) and np.array_equal(decoded["base420"], decoded["prog420"])
    assert np.array_equal(decoded["base444"], decoded["prog444r"]) and np.array_equal(decoded["gray"], decoded["grayprog"])
    luma = smooth @ np.array([0.299, 0.587, 0.114])
    for name, image in decoded.items():
        expected = np.dstack([luma] * 3) if name.startswith("gray") else smooth
        error = np.abs(image.astype(float) - expected).mean()
        assert error < 8, (name, error)
    # Malformed input is rejected with a message.
    good = (root / "rgb.png").read_bytes()
    (root / "truncated.png").write_bytes(good[:len(good) // 2])
    (root / "fake.webp").write_bytes(b"RIFF\x10\x00\x00\x00WEBPVP8 ")
    (root / "noise.bin").write_bytes(bytes(range(256)))
    (root / "truncated.jpg").write_bytes((root / "prog420.jpg").read_bytes()[:150])
    errors = decode(bins, root, [root / n for n in ["truncated.png", "fake.webp", "noise.bin", "truncated.jpg"]])
    assert all(isinstance(e, str) for e in errors), errors
    assert "WebP" in errors[1] and "unrecognized" in errors[2], errors
    print(f"decoding: {len(cases)} PNG/BMP/PPM cases exact, {len(variants)} JPEG variants consistent", flush=True)


def check_tower(root, bins, text, vision, backend, placement):
    image = test_image(150, 110, 2)
    codecs.write_ppm(root / "tower.ppm", image)
    prefix = root / f"tower-{backend}-{placement}"
    result = json.loads(run(bins / "tidal-vision", "--model", text, "--vision-model", vision, "--backend", backend,
                            "--vision-placement", placement, "--images", root / "tower.ppm", "--dump-embeddings", prefix,
                            "--threads", 4).stdout)
    reference = VisionReference(vision)
    grid_w, grid_h, expected = reference.encode(image)
    info = result["images"][0]
    assert info["grid"] == [grid_w, grid_h] and info["tokens"] == grid_w * grid_h, (info, grid_w, grid_h)
    actual = np.fromfile(f"{prefix}.0.f32", dtype=np.float32).reshape(expected.shape)
    error = np.abs(actual - expected).max() / np.abs(expected).max()
    # FlashAttention runs on FP16 K/V.
    assert error < 4e-3, (backend, placement, error)
    assert result["vision"]["placement"] == placement and result["vision"]["device"] == ("CPU" if placement == "cpu" else "CUDA0")
    print(f"vision tower {backend}/{placement}: {grid_w}x{grid_h} tokens, max relative error {error:.2e}", flush=True)


def check_prompt(root, bins, text, vision, backend, label, *flags, ple=False):
    images = {"A": test_image(96, 64, 3), "B": test_image(64, 128, 4)}
    for name, image in images.items():
        codecs.write_ppm(root / f"{name}.ppm", image)
    order = [item for item in PROMPT if isinstance(item, str)]
    tokens = ",".join("IMG" if isinstance(item, str) else str(item) for item in PROMPT)
    prefix = root / f"prompt-{label}"
    trace = root / f"prompt-{label}.json"
    run(bins / "tidal-vision", "--model", text, "--vision-model", vision, "--backend", backend, "--images",
        ",".join(str(root / f"{n}.ppm") for n in order), "--tokens", tokens, "--logits", 1, "--steps", STEPS,
        "--image-min-tokens", 1, "--dump-embeddings", prefix, "--json", trace, "--threads", 4, *flags)
    result = json.loads(trace.read_text())
    run_result = result["runs"][0]
    reference = Reference(text, backend)
    execution = "sequential" if backend == "cpu" else "layer_major" if "--hybrid-batch-tokens" in flags or "gpu" in flags else "hybrid"
    rows = {}
    for i, name in enumerate(order):
        info = result["images"][i]
        width, height = info["grid"]
        rows[name] = (width, height, np.fromfile(f"{prefix}.{i}.f32", dtype=np.float32).reshape(width * height, -1))
    sequence, entries = [], []
    for item in PROMPT:
        if isinstance(item, str):
            width, height, embedding = rows[item]
            base = entries[-1][2][3] if entries else 0
            for j in range(width * height):
                entries.append((IMAGE_TOKEN, embedding[j], (base, base + j // width, base + j % width, base + max(width, height))))
        else:
            t = entries[-1][2][3] if entries else 0
            entries.append((item, None, (t, t, t, t + 1)))
    assert result["prompt_tokens"] == len(entries)
    prompt_length = len(entries)
    tolerance = 2e-3 if backend == "cpu" else 4e-3
    logits = run_result["logits"] + run_result["step_logits"][1:]
    generated = run_result["generated"]
    assert len(run_result["logits"]) == prompt_length
    worst = 0
    for i in range(prompt_length + STEPS - 1):
        if i < prompt_length:
            token, embedding, rotary = entries[i]
        else:
            token, embedding, rotary = generated[i - prompt_length], None, None
        reference.prefill_columns = prompt_length if execution in ("layer_major", "hybrid") and i < prompt_length else 0
        reference.cpu_batch_attention = execution == "hybrid"
        reference.prefill_head_columns = reference.prefill_columns if i != prompt_length - 1 else 0
        expected, _ = reference.decode(token, embedding, rotary)
        actual = np.array(logits[i])
        error = float(np.max(np.abs(actual - expected)) / np.max(np.abs(expected)))
        worst = max(worst, error)
        assert error < tolerance, (label, i, error)
        if i >= prompt_length - 1:
            assert generated[i - prompt_length + 1] == int(np.argmax(expected)), (label, i, "greedy token")
    # Text after the images continues from each image's start + max(rows, columns), not from its index.
    assert reference.rope_table[-1][0] < len(reference.rope_table) - 1
    print(f"prompt {label}: {prompt_length} tokens + {STEPS} steps, max relative logit error {worst:.2e}", flush=True)


def check_cache(root, bins, text, vision, backend):
    image = test_image(96, 96, 5)
    codecs.write_ppm(root / "cache.ppm", image)
    trace = root / "cache.json"
    run(bins / "tidal-vision", "--model", text, "--vision-model", vision, "--backend", backend, "--images", root / "cache.ppm",
        "--tokens", "3,IMG,8,9", "--cached", 1, "--repeat", 2, "--json", trace)
    first, second = json.loads(trace.read_text())["runs"]
    assert first["images_encoded"] == 1 and second["images_encoded"] == 0, (first, second)
    assert second["evaluated"] == 0 and second["reused"] == json.loads(trace.read_text())["prompt_tokens"], second
    assert first["generated"] == second["generated"]
    print("prompt cache: the second identical prompt reused every token without re-encoding", flush=True)


def check_rejections(root, bins, text, vision, backend):
    codecs.write_ppm(root / "x.ppm", test_image(64, 64, 6))
    base = [bins / "tidal-vision", "--model", text, "--backend", backend, "--images", root / "x.ppm"]
    # Placeholder tokens must be the model's image token.
    failure = run(*base, "--vision-model", vision, "--tokens", "1,IMG,2", "--image-token", IMAGE_TOKEN - 1, ok=False)
    assert "placeholder" in failure.stderr, failure.stderr
    # A tower whose output width differs from the text model's hidden size.
    narrow = root / "narrow-vision.gguf"
    run(bins / "tidal-dummy", "--vision", 1, "--output", narrow, "--vision-output", 64)
    failure = run(*base, "--vision-model", narrow, ok=False)
    assert "does not match the text model" in failure.stderr, failure.stderr
    failure = run(bins / "tidal-vision", "--model", text, "--backend", backend, "--vision-model", text, ok=False)
    assert "mmproj" in failure.stderr, failure.stderr
    print("rejections: wrong placeholder token, mismatched tower width, non-mmproj file", flush=True)


def png_data_uri(image):
    path = io.BytesIO()
    target = Path(tempfile.mktemp(suffix=".png"))
    codecs.write_png(target, image, 2)
    data = target.read_bytes()
    target.unlink()
    return "data:image/png;base64," + base64.b64encode(data).decode()


def check_server(root, bins, text, vision, backend):
    server = Server(bins / "tidal-server", "--model", text, "--vision-model", vision, "--backend", backend,
                    "--context", 256, "--image-min-tokens", 1, "--threads", 4)
    try:
        assert server.port, server.output()
        props = server.get("/props")
        assert props["modalities"]["vision"] and props["vision"]["output"] == 128, props
        image = test_image(96, 64, 8)
        uri = png_data_uri(image)
        message = {"role": "user", "content": [{"type": "image_url", "image_url": {"url": uri}}, {"type": "text", "text": "hi"}]}
        body = {"messages": [message], "max_tokens": 4, "temperature": 0}
        # A different prompt restores an empty sequence (dropping later snapshots), so it goes first.
        text_only = server.post("/v1/chat/completions", {**body, "messages": [{"role": "user", "content": "hi"}]})
        first = server.post("/v1/chat/completions", body)
        # The image adds its 3x2 placeholder tokens plus <|vision_start|>/<|vision_end|> rendered as bytes.
        markers = len("<|vision_start|><|vision_end|>")
        assert first["usage"]["prompt_tokens"] == text_only["usage"]["prompt_tokens"] + 6 + markers, (first["usage"], text_only["usage"])
        # Extending the conversation reuses the prefix, image included (the engine reuse; the prompt is
        # below the OpenAI caching minimum, so usage reports no cached tokens).
        reply = first["choices"][0]["message"]["content"]
        turns = [message, {"role": "assistant", "content": reply}, {"role": "user", "content": "more"}]
        before = server.get("/metrics")
        extended = server.post("/v1/chat/completions", {**body, "messages": turns})
        assert extended["timings"]["cache_n"] >= first["usage"]["prompt_tokens"] - 1, extended["timings"]
        after = server.get("/metrics")
        count = lambda text: float(re.search(r"^tidal:images_encoded_total (\S+)$", text, re.M).group(1))
        assert count(after) == count(before), (before, after)
        # A different image of the same size is not mistaken for the cached one.
        other = {**message, "content": [{"type": "image_url", "image_url": {"url": png_data_uri(255 - image)}}, {"type": "text", "text": "hi"}]}
        changed = server.post("/v1/chat/completions", {**body, "messages": [other]})
        assert changed["timings"]["cache_n"] < first["usage"]["prompt_tokens"] - 6, changed["timings"]
        # Anthropic image blocks and llama.cpp multimodal_data.
        data = uri.split(",", 1)[1]
        anthropic = server.post("/v1/messages", {"max_tokens": 4, "temperature": 0, "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}, {"type": "text", "text": "hi"}]}]})
        assert anthropic["usage"]["output_tokens"] >= 1
        native = server.post("/completion", {"prompt": {"prompt_string": "look <__media__> here", "multimodal_data": [data]},
                                             "n_predict": 2, "temperature": 0})
        assert native["tokens_evaluated"] == len("look <|vision_start|>") + 6 + len("<|vision_end|> here"), native["tokens_evaluated"]
        for bad in [{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,!!!"}},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(b"not an image").decode()}}]:
            error = server.post("/v1/chat/completions", {**body, "messages": [{"role": "user", "content": [bad]}]}, 400)["error"]
            assert error["type"] == "invalid_request_error", error
        error = server.post("/v1/chat/completions", {**body, "messages": [{"role": "system", "content": [
            {"type": "image_url", "image_url": {"url": uri}}]}, {"role": "user", "content": "x"}]}, 400)["error"]
        assert "not allowed" in error["message"], error
        placement = server.get("/placement")
        components = {t["component"] for t in placement["tensors"]}
        assert components == {"text", "vision"}, components
        vision_units = {t["compute"] for t in placement["tensors"] if t["component"] == "vision" and t["name"] != "v.position_embd.weight"}
        assert vision_units == {"CPU" if backend == "cpu" else "CUDA0"}, vision_units
        kv = [t for t in placement["tensors"] if t["role"] == "kv_cache"]
        assert kv and all(t["memory"] == "CPU" and t["compute"] == "CPU" for t in kv), kv
        summary = server.get("/placement?tensors=0&component=vision")
        assert "tensors" not in summary and all(s["component"] == "vision" for s in summary["summary"])
        server.get("/placement?component=wizard", 400)
    finally:
        server.process.terminate()
        server.process.wait(timeout=30)
    print("server: image chat (OpenAI, Anthropic, /completion), cache reuse, rejections and /placement", flush=True)


def check_store(root, bins, text, vision, backend):
    """A conversation with an image leaves the engine and comes back from the cache store."""
    server = Server(bins / "tidal-server", "--model", text, "--vision-model", vision, "--backend", backend, "--context", 256,
                    "--image-min-tokens", 1, "--threads", 4, "--cache-ram", "64M", "--cache-chunk-tokens", 16)
    try:
        assert server.port, server.output()
        image = test_image(96, 64, 8)
        with_image = lambda pixels: {"role": "user", "content": [{"type": "image_url", "image_url": {"url": png_data_uri(pixels)}},
                                                                 {"type": "text", "text": "what is in this picture?"}]}
        body = {"max_tokens": 4, "temperature": 0}
        other = {**body, "messages": [{"role": "user", "content": "an unrelated conversation without any image at all"}]}
        count = lambda: float(re.search(r"^tidal:images_encoded_total (\S+)$", server.get("/metrics"), re.M).group(1))
        first = server.post("/v1/chat/completions", {**body, "messages": [with_image(image)]})
        tokens = first["usage"]["prompt_tokens"]
        server.post("/v1/chat/completions", other)
        # Chunks carry image positions (their MRoPE rows and content hashes): the
        # whole prompt comes back without the vision tower running again.
        encoded = count()
        again = server.post("/v1/chat/completions", {**body, "messages": [with_image(image)]})
        assert again["timings"]["cache_n"] == tokens and again["timings"]["cache_store_n"] == tokens, again["timings"]
        assert count() == encoded and again["choices"][0]["message"] == first["choices"][0]["message"], (first, again)
        # The same tokens with another image are another prefix.
        server.post("/v1/chat/completions", other)
        changed = server.post("/v1/chat/completions", {**body, "messages": [with_image(255 - image)]})
        assert changed["timings"]["cache_store_n"] == 0 and changed["timings"]["cache_n"] < tokens - 6, changed["timings"]
    finally:
        server.process.terminate()
        server.process.wait(timeout=30)
    print("server: image conversation restored from the cache store", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    bins, backend = args.bin_dir, args.backend
    failures = []
    with tempfile.TemporaryDirectory(prefix="tidal-vision-") as tmp:
        root = Path(tmp)
        text, ple, vision = root / "text.gguf", root / "ple.gguf", root / "vision.gguf"
        run(bins / "tidal-dummy", "--output", text, "--context", 512, "--image-token", IMAGE_TOKEN)
        run(bins / "tidal-dummy", "--output", ple, "--context", 512, "--image-token", IMAGE_TOKEN, "--ple-layers", "0,2")
        run(bins / "tidal-dummy", "--vision", 1, "--output", vision, "--vision-hidden", 128, "--vision-heads", 2)
        cases = [("decoding", lambda: check_decoding(root, bins)),
                 ("tower-cpu", lambda: check_tower(root, bins, text, vision, backend, "cpu"))]
        if backend == "cuda":
            cases += [("tower-gpu", lambda: check_tower(root, bins, text, vision, backend, "gpu"))]
            cases += [("prompt-hybrid", lambda: check_prompt(root, bins, text, vision, backend, "hybrid")),
                      ("prompt-layer-major", lambda: check_prompt(root, bins, text, vision, backend, "layer-major", "--hybrid-batch-tokens", 0)),
                      ("prompt-gpu-kv", lambda: check_prompt(root, bins, text, vision, backend, "gpu-kv", "--kv-placement", "gpu")),
                      ("prompt-ple", lambda: check_prompt(root, bins, ple, vision, backend, "ple-hybrid")),
                      ("prompt-ple-layer-major", lambda: check_prompt(root, bins, ple, vision, backend, "ple-layer-major", "--hybrid-batch-tokens", 0))]
        else:
            cases += [("prompt-sequential", lambda: check_prompt(root, bins, text, vision, backend, "sequential")),
                      ("prompt-ple", lambda: check_prompt(root, bins, ple, vision, backend, "ple-sequential"))]
        cases += [("cache", lambda: check_cache(root, bins, text, vision, backend)),
                  ("rejections", lambda: check_rejections(root, bins, text, vision, backend)),
                  ("server", lambda: check_server(root, bins, text, vision, backend)),
                  ("store", lambda: check_store(root, bins, text, vision, backend))]
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
