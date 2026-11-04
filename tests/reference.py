"""Independent NumPy oracle for the documented tiny Dense model.

Reads GGUF and dequantizes its weights without binding to tidal or ggml. Equations
follow Transformers' Qwen4Exp GDN, QSA and GatedResidual (see docs/architecture.md).
"""
from __future__ import annotations

import struct
from pathlib import Path
import numpy as np


def bf16(x):
    x = np.asarray(x, dtype=np.float32)
    bits = x.view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000).view(np.float32)


def read_gguf(path, quantized=None):
    data = Path(path).read_bytes()
    position = 0

    def take(fmt):
        nonlocal position
        result = struct.unpack_from("<" + fmt, data, position)
        position += struct.calcsize("<" + fmt)
        return result[0] if len(result) == 1 else result

    def string():
        nonlocal position
        length = take("Q")
        text = data[position:position + length].decode()
        position += length
        return text

    assert data[:4] == b"GGUF"
    position = 4
    assert take("I") == 3
    n_tensors, n_metadata = take("QQ")
    metadata = {}
    metadata_offsets = {}
    for _ in range(n_metadata):
        key, kind = string(), take("I")
        metadata_offsets[key] = position
        if kind == 9:
            element_kind, count = take("I"), take("Q")
            metadata[key] = [take({4: "I", 10: "Q"}[element_kind]) for _ in range(count)]
        else:
            metadata[key] = string() if kind == 8 else take({4: "I", 5: "i", 6: "f", 7: "?", 10: "Q"}[kind])
    tensors = []
    for _ in range(n_tensors):
        name, dimensions = string(), take("I")
        shape = [take("Q") for _ in range(dimensions)]
        kind, offset = take("I"), take("Q")
        tensors.append((name, shape, kind, offset))
    alignment = metadata.get("general.alignment", 32)
    start = (position + alignment - 1) // alignment * alignment
    weights, types = {}, {}
    for name, shape, kind, offset in tensors:
        offset += start
        count = int(np.prod(shape))
        if kind in (0, 1):
            value = np.frombuffer(data, dtype="<f4" if kind == 0 else "<f2", count=count, offset=offset).astype(np.float32)
        elif kind == 30:
            value = (np.frombuffer(data, dtype="<u2", count=count, offset=offset).astype(np.uint32) << 16).view(np.float32)
        elif kind in (2, 8):
            block_bytes = 18 if kind == 2 else 34
            raw = np.frombuffer(data, dtype=np.uint8, count=count // 32 * block_bytes, offset=offset).reshape(-1, block_bytes)
            scale = raw[:, :2].copy().view("<f2").astype(np.float32)
            if kind == 2:
                quant = np.concatenate([raw[:, 2:] & 15, raw[:, 2:] >> 4], axis=1).astype(np.float32) - 8
            else:
                quant = raw[:, 2:].copy().view(np.int8).astype(np.float32)
            value = (scale * quant).reshape(-1)
            if quantized is not None:
                # Keep block scales and integer codes: native quantized dot
                # products scale integer sums, not individual float products.
                quantized[name] = (scale.reshape(-1, shape[0] // 32),
                                   quant.reshape(-1, shape[0] // 32, 32))
        else:
            raise ValueError(f"Unsupported GGUF type: {kind}")
        weights[name] = value.reshape(tuple(reversed(shape)))
        types[name] = kind
    config = {key[6:]: value for key, value in metadata.items() if key.startswith("tidal.")}
    return config, weights, types, metadata_offsets


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def silu(x):
    return x * sigmoid(x)


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class Reference:
    def __init__(self, path, backend="cpu"):
        if backend not in ("cpu", "cuda"):
            raise ValueError(f"Unknown reference backend: {backend}")
        self.backend = backend
        self.quantized = {}
        self.c, self.w, self.types, _ = read_gguf(path, self.quantized)
        # GGUF omits trailing dimensions of length one. Preserve the logical
        # expert axis even for the degenerate one-expert configuration.
        for name, weight in self.w.items():
            if name.endswith("_exps") and weight.ndim == 2:
                self.w[name] = weight[None, ...]
            elif name.endswith(".ffn.router") and weight.ndim == 1:
                self.w[name] = weight[None, ...]
        c = self.c
        self.position = 0
        self.tokens = []
        self.states = []
        for layer in range(c["layers"]):
            if (layer + 1) % 4:
                channels = (2 * c["gdn_key_heads"] + c["gdn_value_heads"]) * c["gdn_dim"]
                self.states.append({
                    "conv": np.zeros((channels, c["conv_kernel"] - 1), dtype=np.float32),
                    "recurrent": np.zeros((c["gdn_value_heads"], c["gdn_dim"], c["gdn_dim"]), dtype=np.float32),
                })
            else:
                self.states.append({"keys": [], "values": [], "raw": [], "pooled": []})
            if layer in c.get("ple.layers", []):
                history = (c["ple.conv_kernel"] - 1) * c["ple.ngram_size"]
                self.states[-1]["ple_conv"] = np.zeros((history, c["branches"], c["hidden"]), dtype=np.float32)

    def mm(self, name, x, expert=None):
        # Model the pinned backend's single-vector arithmetic independently.
        # CUDA F16/BF16 matvecs with F32 accumulation keep F32 activations;
        # CPU dot products convert them to the weight's floating-point type.
        kind = self.types[name]
        x = np.asarray(x, dtype=np.float32)
        if kind == 1 and (self.backend == "cpu" or expert is not None):
            x = x.astype(np.float16).astype(np.float32)
        elif kind == 30 and (self.backend == "cpu" or expert is not None):
            x = bf16(x)
        elif kind in (2, 8):
            blocks = x.reshape(-1, 32)
            maximum = np.max(np.abs(blocks), axis=-1)
            scale = maximum / 127
            if self.backend == "cpu":
                # AVX2 Q8_0 activation quantization: nearest-even, 127/max.
                inv = np.divide(127, maximum, out=np.zeros_like(maximum), where=maximum != 0)
                quant = np.rint(blocks * inv[:, None])
            else:
                # CUDA Q8_1 uses roundf(x/d), with halfway cases away from zero.
                # ggml enables fast math: division becomes reciprocal multiply.
                # Quantized PLE rows frequently hit the resulting halfway cases.
                scale = maximum * np.float32(1 / 127)
                inverse = np.divide(1, scale, out=np.zeros_like(scale), where=scale != 0)
                scaled = blocks * inverse[:, None]
                magnitude = np.abs(scaled)
                integral = np.floor(magnitude)
                # Adding 0.5 in F32 can round a value just BELOW a tie upward.
                quant = np.copysign(integral + (magnitude - integral >= 0.5), scaled)
            scale = scale.astype(np.float16).astype(np.float32)
            weight_scale, codes = self.quantized[name]
            if expert is not None:
                rows = self.w[name].shape[1]
                weight_scale = weight_scale[expert * rows:(expert + 1) * rows]
                codes = codes[expert * rows:(expert + 1) * rows]
            if kind == 2 and self.backend == "cuda":
                # Q4_0 stores unsigned nibbles. CUDA subtracts their zero point
                # using Q8_1's FP16 sum of ORIGINAL activations, not sum(q)*d.
                # These differ even when no quantization tie is involved.
                dot = np.einsum("obd,bd->ob", codes + 8, quant)
                original_sum = blocks.sum(axis=-1).astype(np.float16).astype(np.float32)
                return (weight_scale * (dot * scale - 8 * original_sum)).sum(axis=-1)
            dot = np.einsum("obd,bd->ob", codes, quant)
            return (weight_scale * scale * dot).sum(axis=-1)
        return (self.w[name] if expert is None else self.w[name][expert]) @ x

    def norm(self, name, x):
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + self.c["eps"]) * self.w[name]

    def rope(self, x, position):
        n = self.c["rope_dim"]
        theta = position / (np.float32(self.c["rope_base"]) ** (np.arange(n // 2, dtype=np.float32) * 2 / n))
        x = x.copy()
        left, right = x[..., :n // 2].copy(), x[..., n // 2:n].copy()
        x[..., :n // 2] = left * np.cos(theta) - right * np.sin(theta)
        x[..., n // 2:n] = right * np.cos(theta) + left * np.sin(theta)
        return x

    def mix(self, prefix, h, inject=True):
        x = self.norm(prefix + ".norm", h)
        flat = x.reshape(-1)
        gate = sigmoid(self.mm(prefix + ".up", silu(self.mm(prefix + ".down", flat) / self.c["branches"])))
        mixed = (gate.reshape(x.shape) * x).mean(axis=0)
        injection = 2 * sigmoid(self.mm(prefix + ".inject", flat) / self.c["branches"]) if inject else None
        return mixed, injection

    def ffn(self, prefix, h):
        x, injection = self.mix(prefix + "hc_ffn", h)
        p = prefix + "ffn."
        if self.c.get("experts", 0):
            probabilities = softmax(self.mm(p + "router", x))
            ids = np.argsort(-probabilities, kind="stable")[:self.c["experts_used"]]
            scores = probabilities[ids]
            if self.c["norm_topk"]:
                scores = scores / scores.sum()
            output = np.zeros(self.c["hidden"], dtype=np.float32)
            for expert, score in zip(ids, scores):
                active = silu(self.mm(p + "gate_exps", x, expert)) * self.mm(p + "up_exps", x, expert)
                output += score * self.mm(p + "down_exps", active, expert)
            if self.c["shared_ffn"]:
                shared = self.mm(p + "shared.down", silu(self.mm(p + "shared.gate", x)) * self.mm(p + "shared.up", x))
                output += sigmoid(self.mm(p + "shared_gate", x)) * shared
        else:
            output = self.mm(p + "down", silu(self.mm(p + "gate", x)) * self.mm(p + "up", x))
        return h + injection[:, None] * output

    def gdn(self, p, x, state):
        c = self.c
        d, hk, hv = c["gdn_dim"], c["gdn_key_heads"], c["gdn_value_heads"]
        projected = self.mm(p + "gdn.qkv", x)
        history = np.concatenate([state["conv"], projected[:, None]], axis=1)
        state["conv"] = history[:, 1:].copy()
        conv = silu(np.sum(history * self.w[p + "gdn.conv"], axis=1))
        q, k, v = np.split(conv, [d * hk, 2 * d * hk])
        q, k = q.reshape(hk, d), k.reshape(hk, d)
        q = q / np.sqrt((q * q).sum(axis=-1, keepdims=True) + c["eps"])
        k = k / np.sqrt((k * k).sum(axis=-1, keepdims=True) + c["eps"])
        q, k = np.repeat(q, hv // hk, axis=0), np.repeat(k, hv // hk, axis=0)
        v = v.reshape(hv, d)
        a = self.mm(p + "gdn.a", x) + self.w[p + "gdn.dt_bias"].reshape(-1)
        decay = -np.exp(self.w[p + "gdn.a_log"].reshape(-1)) * np.logaddexp(0, a)
        beta = sigmoid(self.mm(p + "gdn.b", x))
        recurrent = state["recurrent"] * np.exp(decay)[:, None, None]
        prediction = np.einsum("hkv,hk->hv", recurrent, k)
        delta = beta[:, None] * (v - prediction)
        recurrent += k[:, :, None] * delta[:, None, :]
        state["recurrent"] = recurrent
        output = np.einsum("hkv,hk->hv", recurrent, q / np.sqrt(np.float32(d)))
        z = self.mm(p + "gdn.z", x).reshape(hv, d)
        output = self.norm(p + "gdn.norm", output) * silu(z)
        return self.mm(p + "gdn.out", output.reshape(-1))

    def ple(self, layer, token, h, state):
        c = self.c
        p, meta = f"blk.{layer}.ple.", f"ple.{layer}."
        # Construct the whole segment independently of the engine's bounded
        # window. The current EOS retains its context; the following token resets.
        last_eos = max((i for i, t in enumerate(self.tokens) if t == c["ple.eos_token_id"]), default=-1)
        segment = self.tokens[last_eos + 1:] + [token]
        shifted = [segment[-1 - j] if j < len(segment) else c["ple.eos_token_id"]
                   for j in range(c["ple.ngram_size"])]
        rows = []
        for n in range(2, c["ple.ngram_size"] + 1):
            mixed = 0
            for j in range(n):
                mixed ^= shifted[j] * c[meta + "multipliers"][j]
            for head in range(c["ple.heads_per_ngram"]):
                i = (n - 2) * c["ple.heads_per_ngram"] + head
                rows.append(mixed % c[meta + "vocab_sizes"][i] + c[meta + "offsets"][i])
        emb = self.w[p + "embedding"][rows].reshape(-1)
        key = self.norm(p + "norm_key", self.mm(p + "key", emb).reshape(h.shape))
        query = self.norm(p + "norm_query", h)
        score = (key * query).sum(axis=-1, keepdims=True) / np.sqrt(np.float32(c["hidden"]))
        gate = sigmoid(np.sign(score) * np.sqrt(np.maximum(np.abs(score), np.float32(1e-6))))
        gated = gate * self.mm(p + "value", emb)
        normalized = self.norm(p + "norm_conv", gated)
        history = np.concatenate([state["ple_conv"], normalized[None]], axis=0)
        state["ple_conv"] = history[1:].copy()
        taps = history[::c["ple.ngram_size"]]
        kernel = self.w[p + "conv"].reshape(c["branches"], c["hidden"], c["ple.conv_kernel"])
        convolved = np.einsum("tbh,bht->bh", taps, kernel)
        return h + (gated + silu(convolved))

    def attention(self, p, x, state):
        c, pos = self.c, self.position
        d, h, hk, di = c["head_dim"], c["q_heads"], c["kv_heads"], c["index_dim"]
        qg = self.mm(p + "attn.qg", x).reshape(h, 2 * d)
        q = self.rope(self.norm(p + "attn.q_norm", qg[:, :d]), pos)
        k = self.rope(self.norm(p + "attn.k_norm", self.mm(p + "attn.k", x).reshape(hk, d)), pos)
        v = self.mm(p + "attn.v", x).reshape(hk, d)
        state["keys"].append(bf16(k))
        state["values"].append(bf16(v))
        iq = self.rope(self.norm(p + "index.q_norm", self.mm(p + "index.q", x).reshape(c["index_heads"], di)), pos)
        state["raw"].append(self.mm(p + "index.k", x))
        block = c["block_size"]
        if (pos + 1) % block == 0:
            pooled = np.mean(state["raw"][-block:], axis=0)
            pooled = self.rope(self.norm(p + "index.k_norm", pooled), pos + 1 - block)
            state["pooled"].append(bf16(pooled.reshape(-1)))
        n = (pos + 1) // block
        selected = []
        if n:
            scores = np.maximum(iq @ np.stack(state["pooled"]).T, 0).sum(axis=0) / np.sqrt(np.float32(di))
            chosen = np.argsort(-scores, kind="stable")[:c["top_blocks"]]
            selected = [int(b) * block + j for b in chosen for j in range(block)]
        selected += list(range(n * block, pos + 1))
        selected.sort()
        keys = np.stack(state["keys"])[selected]
        values = np.stack(state["values"])[selected]
        keys, values = np.repeat(keys, h // hk, axis=1), np.repeat(values, h // hk, axis=1)
        scores = np.einsum("hd,thd->ht", q, keys) / np.sqrt(np.float32(d))
        output = np.einsum("ht,thd->hd", softmax(scores), values)
        output *= sigmoid(qg[:, d:])
        return self.mm(p + "attn.out", output.reshape(-1)), selected

    def decode(self, token):
        c = self.c
        h = np.tile(self.w["token_embd.weight"][token], (c["branches"], 1))
        selections = []
        for layer, state in enumerate(self.states):
            p = f"blk.{layer}."
            if layer in c.get("ple.layers", []):
                h = self.ple(layer, token, h, state)
            x, injection = self.mix(p + "hc_attn", h)
            if (layer + 1) % 4:
                output = self.gdn(p, x, state)
            else:
                output, selected = self.attention(p, x, state)
                selections.append({"layer": layer, "tokens": selected})
            h = self.ffn(p, h + injection[:, None] * output)
        x, _ = self.mix("head", h, inject=False)
        self.position += 1
        self.tokens.append(token)
        return self.mm("output.weight", x), selections
