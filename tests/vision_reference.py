"""Independent NumPy oracle for the Qwen3-VL vision tower (mmproj GGUF) and
Qwen-VL image preprocessing: smart resize, Pillow bicubic resampling,
normalization, 2x2 merged patch order, learned position interpolation,
2D rotary embedding, pre-norm ViT blocks and the patch merger.
"""
from __future__ import annotations

import math

import numpy as np

from reference import read_gguf


def smart_resize(width, height, factor, min_pixels, max_pixels):
    """F32 arithmetic and half-to-even rounding, as the engine and llama.cpp."""
    f32 = np.float32
    w = max(factor, int(np.rint(f32(width) / f32(factor))) * factor)
    h = max(factor, int(np.rint(f32(height) / f32(factor))) * factor)
    if w * h > max_pixels:
        beta = np.sqrt(f32(height) * f32(width) / f32(max_pixels))
        h = max(factor, int(np.floor(f32(height) / beta / f32(factor))) * factor)
        w = max(factor, int(np.floor(f32(width) / beta / f32(factor))) * factor)
    elif w * h < min_pixels:
        beta = np.sqrt(f32(min_pixels) / (f32(height) * f32(width)))
        h = int(np.ceil(f32(height) * beta / f32(factor))) * factor
        w = int(np.ceil(f32(width) * beta / f32(factor))) * factor
    return w, h


def _bicubic(x):
    a = -0.5
    x = abs(x)
    if x < 1:
        return ((a + 2) * x - (a + 3)) * x * x + 1
    if x < 2:
        return (((x - 5) * x + 8) * x - 4) * a
    return 0.0


def _coefficients(size_in, size_out):
    scale = size_in / size_out
    filter_scale = max(1.0, scale)
    support = 2.0 * filter_scale
    rows = []
    for o in range(size_out):
        center = (o + 0.5) * scale
        first = max(0, int(center - support + 0.5))
        count = min(size_in, int(center + support + 0.5)) - first
        weights = [_bicubic((i + first - center + 0.5) / filter_scale) for i in range(count)]
        total = sum(weights)
        weights = [w / total if total else w for w in weights]
        # Pillow: 22-bit fixed point, +-0.5 then truncation toward zero.
        rows.append((first, [int(w * (1 << 22) + (-0.5 if w < 0 else 0.5)) for w in weights]))
    return rows


def resize_bicubic(rgb, width, height):
    """Pillow Image.resize(BICUBIC) for RGB8: horizontal, then vertical pass."""
    image = np.asarray(rgb, dtype=np.int64)

    def one_axis(data, size_out, axis):
        size_in = data.shape[axis]
        if size_in == size_out:
            return data
        moved = np.moveaxis(data, axis, 0)
        out = np.empty((size_out,) + moved.shape[1:], dtype=np.int64)
        for o, (first, weights) in enumerate(_coefficients(size_in, size_out)):
            acc = np.full(moved.shape[1:], 1 << 21, dtype=np.int64)
            for i, w in enumerate(weights):
                acc += moved[first + i] * w
            out[o] = np.clip(acc >> 22, 0, 255)
        return np.moveaxis(out, 0, axis)

    return one_axis(one_axis(image, width, 1), height, 0).astype(np.uint8)


def _gelu_tanh(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def _gelu_erf(x):
    return 0.5 * x * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))


def _layer_norm(x, weight, bias, eps):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight + bias


class VisionReference:
    def __init__(self, path, min_tokens=8, max_tokens=4096):
        _, self.w, _, _ = read_gguf(path)
        from gguf_metadata import read_header
        self.meta = read_header(path)[0]
        m = self.meta
        assert m["clip.projector_type"] == "qwen3vl_merger"
        self.hidden = m["clip.vision.embedding_length"]
        self.heads = m["clip.vision.attention.head_count"]
        self.layers = m["clip.vision.block_count"]
        self.patch = m["clip.vision.patch_size"]
        self.merge = m.get("clip.vision.spatial_merge_size", 2)
        self.eps = m.get("clip.vision.attention.layer_norm_epsilon", 1e-6)
        self.mean = np.array(m.get("clip.vision.image_mean", [0.5] * 3), dtype=np.float64)
        self.std = np.array(m.get("clip.vision.image_std", [0.5] * 3), dtype=np.float64)
        self.min_tokens, self.max_tokens = min_tokens, max_tokens
        self.side = int(round(math.sqrt(self.w["v.position_embd.weight"].shape[0])))
        self.w = {k: v.astype(np.float64) for k, v in self.w.items()}

    def resized(self, width, height):
        factor = self.patch * self.merge
        return smart_resize(width, height, factor, self.min_tokens * factor * factor, self.max_tokens * factor * factor)

    def encode(self, rgb):
        """rgb: (H, W, 3) uint8. Returns (grid_w, grid_h, embeddings[tokens, out])."""
        height, width = rgb.shape[:2]
        w_out, h_out = self.resized(width, height)
        image = resize_bicubic(rgb, w_out, h_out).astype(np.float64)
        image = ((image / 255.0) - self.mean) / self.std
        p, hidden = self.patch, self.hidden
        ph, pw = h_out // p, w_out // p
        # Patches in merge order (2x2 neighbours consecutive), vectors (c, y, x).
        order = [(2 * ty + dy, 2 * tx + dx) for ty in range(ph // 2) for tx in range(pw // 2) for dy in range(2) for dx in range(2)]
        patches = np.stack([image[y * p:(y + 1) * p, x * p:(x + 1) * p].transpose(2, 0, 1).reshape(-1) for y, x in order])
        kernel = (self.w["v.patch_embd.weight"] + self.w["v.patch_embd.weight.1"]).reshape(hidden, -1)
        x = patches @ kernel.T + self.w["v.patch_embd.bias"]
        # Learned positions: bilinear over the side x side table, aligned corners.
        table = self.w["v.position_embd.weight"].reshape(self.side, self.side, hidden)

        def axis(count):
            out = []
            for i in range(count):
                position = i * (self.side - 1) / (count - 1) if count > 1 else 0.0
                low = min(self.side - 1, int(math.floor(position)))
                out.append((low, min(self.side - 1, low + 1), position - low))
            return out

        rows, cols = axis(ph), axis(pw)
        positions = []
        for y, xx in order:
            y0, y1, fy = rows[y]
            x0, x1, fx = cols[xx]
            positions.append(table[y0, x0] * (1 - fy) * (1 - fx) + table[y0, x1] * (1 - fy) * fx +
                             table[y1, x0] * fy * (1 - fx) + table[y1, x1] * fy * fx)
        x = x + np.stack(positions)
        # 2D rotary embedding: pairs (j, j + d/2); the first d/4 frequencies
        # follow the patch row, the next d/4 its column.
        d = hidden // self.heads
        quarter = d // 4
        inv = 10000.0 ** (-np.arange(quarter) * 2 / (d // 2))
        ys = np.array([y for y, _ in order], dtype=np.float64)
        xs = np.array([xx for _, xx in order], dtype=np.float64)
        angles = np.concatenate([ys[:, None] * inv, xs[:, None] * inv], axis=1)  # [n, d/2]
        cos, sin = np.cos(angles)[:, None, :], np.sin(angles)[:, None, :]

        def rotate(t):
            left, right = t[..., :d // 2], t[..., d // 2:]
            return np.concatenate([left * cos - right * sin, right * cos + left * sin], axis=-1)

        n = len(order)
        for layer in range(self.layers):
            pre = f"v.blk.{layer}."
            h = _layer_norm(x, self.w[pre + "ln1.weight"], self.w[pre + "ln1.bias"], self.eps)
            qkv = h @ self.w[pre + "attn_qkv.weight"].T + self.w[pre + "attn_qkv.bias"]
            q, k, v = (qkv[:, i * hidden:(i + 1) * hidden].reshape(n, self.heads, d) for i in range(3))
            q, k = rotate(q), rotate(k)
            scores = np.einsum("qhd,khd->hqk", q, k) / math.sqrt(d)
            scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
            scores /= scores.sum(axis=-1, keepdims=True)
            attended = np.einsum("hqk,khd->qhd", scores, v).reshape(n, hidden)
            x = x + attended @ self.w[pre + "attn_out.weight"].T + self.w[pre + "attn_out.bias"]
            h = _layer_norm(x, self.w[pre + "ln2.weight"], self.w[pre + "ln2.bias"], self.eps)
            up = _gelu_tanh(h @ self.w[pre + "ffn_up.weight"].T + self.w[pre + "ffn_up.bias"])
            x = x + up @ self.w[pre + "ffn_down.weight"].T + self.w[pre + "ffn_down.bias"]
        x = _layer_norm(x, self.w["v.post_ln.weight"], self.w["v.post_ln.bias"], self.eps).reshape(n // 4, 4 * hidden)
        x = _gelu_erf(x @ self.w["mm.0.weight"].T + self.w["mm.0.bias"])
        x = x @ self.w["mm.2.weight"].T + self.w["mm.2.bias"]
        return pw // 2, ph // 2, x.astype(np.float32)
