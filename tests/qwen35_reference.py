"""Independent NumPy text decoder for native qwen35 GGUFs (F32/F16/BF16/Q8_0).

Uses the published equations, including GGUF's folded norm gains, negative
exponentiated A and tiled V-head ordering. No tidal/ggml inference bindings.
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
from gguf_metadata import read_header
from reference import bf16, sigmoid, silu, softmax, flash_attention_reference


class Qwen35Reference:
    def __init__(self, path, backend="cuda", kv_type="bf16"):
        assert kv_type in ('fp16', 'bf16')
        self.kv_round = bf16 if kv_type == 'bf16' else lambda x: x.astype(np.float16).astype(np.float32)
        self.meta, self.tensors, self.offset = read_header(path)
        assert self.meta["general.architecture"] == "qwen35"
        self.c = {k[7:]: v for k, v in self.meta.items() if k.startswith("qwen35.")}
        self.backend = backend
        self.prefill_columns = 0
        self.gpu_attention = False
        self.data = np.memmap(path, dtype=np.uint8, mode="r")
        self.weights = {}
        self.layers = self.c["block_count"] - self.c.get("nextn_predict_layers", 0)
        self.recurrent = self.c.get("attention.recurrent_layers", [
            (i+1) % self.c.get("full_attention_interval", 4) != 0 for i in range(self.layers)])[:self.layers]
        self.reset()

    def reset(self):
        c = self.c
        hk, hv, d = c["ssm.group_count"], c["ssm.time_step_rank"], c["ssm.state_size"]
        self.states = [dict(conv=np.zeros(((2*hk+hv)*d, c["ssm.conv_kernel"]-1), np.float32),
                            recurrent=np.zeros((hv, d, d), np.float32)) if recurrent else dict(keys=[], values=[])
                       for recurrent in self.recurrent]
        self.position = 0

    def raw(self, name):
        t = self.tensors[name]
        return self.offset + t["offset"], t["type"], tuple(reversed(t["shape"]))

    def q8(self, name):
        offset, kind, shape = self.raw(name)
        assert kind == 8
        blocks = np.ndarray((int(np.prod(shape))//32, 34), dtype=np.uint8, buffer=self.data, offset=offset)
        scales = blocks[:, :2].copy().view("<f2").astype(np.float32).reshape(-1, shape[-1]//32)
        codes = blocks[:, 2:].view(np.int8).reshape(-1, shape[-1]//32, 32)
        return scales, codes

    def weight(self, name):
        if name not in self.weights:
            offset, kind, shape = self.raw(name)
            if kind in (0, 1):
                value = np.ndarray(shape, dtype="<f4" if kind == 0 else "<f2", buffer=self.data, offset=offset).astype(np.float32)
            elif kind == 30:
                bits = np.ndarray(shape, dtype="<u2", buffer=self.data, offset=offset).astype(np.uint32) << 16
                value = bits.view(np.float32)
            elif kind == 8:
                scales, codes = self.q8(name)
                value = (scales[:, :, None] * codes).reshape(shape)
            else:
                raise ValueError(f"Independent oracle does not decode weight type {kind}")
            self.weights[name] = value
        return self.weights[name]

    def mm(self, name, x):
        kind = self.tensors[name]["type"]
        if kind == 30 and self.backend == "cpu": x = bf16(x)
        if kind == 1 and self.backend == "cpu": x = x.astype(np.float16).astype(np.float32)
        if kind == 8:
            columns = self.prefill_columns
            if name in ("output.weight", "token_embd.weight"):
                columns = getattr(self, "prefill_head_columns", columns)
            mmq = self.backend == "cuda" and columns > 8
            blocks = x.reshape(-1, 32)
            maximum = np.max(np.abs(blocks), axis=1)
            if self.backend == "cpu":
                scale = maximum / np.float32(127)
                inv = np.divide(127, maximum, out=np.zeros_like(maximum), where=maximum != 0)
                quant = np.rint(blocks * inv[:, None])
            elif mmq:
                inv = np.divide(127, maximum, out=np.zeros_like(maximum), where=maximum != 0)
                value = blocks * inv[:, None]
                magnitude = np.abs(value)
                lower = np.floor(magnitude)
                quant = np.copysign(lower + (magnitude-lower >= .5), value)
                scale = np.divide(1, inv, out=np.zeros_like(inv), where=inv != 0)
            else:
                scale = maximum * np.float32(1/127)
                inv = np.divide(1, scale, out=np.zeros_like(scale), where=scale != 0)
                value = blocks * inv[:, None]
                magnitude = np.abs(value)
                lower = np.floor(magnitude)
                quant = np.copysign(lower + (magnitude-lower >= .5), value)
            scales, codes = self.q8(name)
            dot = np.einsum("obd,bd->ob", codes, quant)
            if not mmq: scale = scale.astype(np.float16).astype(np.float32)
            return (dot * scales * scale).sum(axis=1)
        return self.weight(name) @ x

    def norm(self, x, name):
        return x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + self.c["attention.layer_norm_rms_epsilon"]) * self.weight(name)

    def rope(self, x):
        n = self.c["rope.dimension_count"]
        theta = np.float32(self.position) / (np.float32(self.c["rope.freq_base"]) ** (np.arange(n//2, dtype=np.float32)*2/n))
        a, b = x[..., :n//2].copy(), x[..., n//2:n].copy()
        x = x.copy()
        x[..., :n//2] = a*np.cos(theta) - b*np.sin(theta)
        x[..., n//2:n] = a*np.sin(theta) + b*np.cos(theta)
        return x

    def gdn(self, p, x, state):
        c = self.c
        hk, hv, d = c["ssm.group_count"], c["ssm.time_step_rank"], c["ssm.state_size"]
        history = np.concatenate([state["conv"], self.mm(p+"attn_qkv.weight", x)[:, None]], axis=1)
        state["conv"] = history[:, 1:].copy()
        conv = silu(np.sum(history * self.weight(p+"ssm_conv1d.weight"), axis=1))
        q, k, v = np.split(conv, [d*hk, 2*d*hk])
        q, k = q.reshape(hk, d), k.reshape(hk, d)
        q /= np.sqrt((q*q).sum(axis=1, keepdims=True) + np.float32(1e-6))
        k /= np.sqrt((k*k).sum(axis=1, keepdims=True) + np.float32(1e-6))
        q, k = np.tile(q, (hv//hk, 1)), np.tile(k, (hv//hk, 1))
        a = self.mm(p+"ssm_alpha.weight", x) + self.weight(p+"ssm_dt.bias")
        decay = self.weight(p+"ssm_a") * np.logaddexp(0, a)
        beta = sigmoid(self.mm(p+"ssm_beta.weight", x))
        state["recurrent"] *= np.exp(decay)[:, None, None]
        predicted = np.einsum("hkv,hk->hv", state["recurrent"], k)
        delta = beta[:, None] * (v.reshape(hv, d) - predicted)
        state["recurrent"] += k[:, :, None] * delta[:, None, :]
        out = np.einsum("hkv,hk->hv", state["recurrent"], q / np.sqrt(np.float32(d)))
        z = self.mm(p+"attn_gate.weight", x).reshape(hv, d)
        return self.mm(p+"ssm_out.weight", (self.norm(out, p+"ssm_norm.weight") * silu(z)).reshape(-1))

    def attention(self, p, x, state):
        c = self.c
        h, hk, d = c["attention.head_count"], c["attention.head_count_kv"], c["attention.key_length"]
        qg = self.mm(p+"attn_q.weight", x).reshape(h, 2*d)
        q = self.rope(self.norm(qg[:, :d], p+"attn_q_norm.weight"))
        k = self.rope(self.norm(self.mm(p+"attn_k.weight", x).reshape(hk, d), p+"attn_k_norm.weight"))
        state["keys"].append(self.kv_round(k))
        state["values"].append(self.kv_round(self.mm(p+"attn_v.weight", x).reshape(hk, d)))
        keys = np.repeat(np.stack(state["keys"]), h//hk, axis=1)
        values = np.repeat(np.stack(state["values"]), h//hk, axis=1)
        if self.backend == "cuda" and self.prefill_columns > 8 and not self.gpu_attention:
            q = q.astype(np.float16).astype(np.float32)
            keys = keys.astype(np.float16).astype(np.float32)
            values = values.astype(np.float16).astype(np.float32)
        if self.gpu_attention:
            attended = flash_attention_reference(q, keys, values, 1/np.sqrt(np.float32(d)))
        else:
            scores = np.einsum("hd,thd->ht", q, keys) / np.sqrt(np.float32(d))
            attended = np.einsum("ht,thd->hd", softmax(scores), values)
        attended *= sigmoid(qg[:, d:])
        return self.mm(p+"attn_output.weight", attended.reshape(-1))

    def decode(self, token):
        h = self.weight("token_embd.weight")[token].copy()
        selections = []
        for i, state in enumerate(self.states):
            p = f"blk.{i}."
            x = self.norm(h, p+"attn_norm.weight")
            if self.recurrent[i]: h += self.gdn(p, x, state)
            else:
                h += self.attention(p, x, state)
                selections.append(dict(layer=i, tokens=list(range(self.position+1))))
            x = self.norm(h, p+"post_attention_norm.weight")
            h += self.mm(p+"ffn_down.weight", silu(self.mm(p+"ffn_gate.weight", x)) * self.mm(p+"ffn_up.weight", x))
        self.position += 1
        output = "output.weight" if "output.weight" in self.tensors else "token_embd.weight"
        return self.mm(output, self.norm(h, "output_norm.weight")), selections
