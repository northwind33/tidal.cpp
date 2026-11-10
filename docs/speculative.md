# Speculative decoding and MTP

With CUDA and CPU KV, `Options::max_draft_tokens > 0` enables speculative
decoding. `Engine::verify(token, drafts)` consumes `token` and checks up to
`max_draft_tokens` drafted continuations in one [hybrid batch](prefill.md#hybrid-batches).
`Engine::draft(token, count)` produces those drafts from the model's MTP block
(`Options::mtp`).

```cpp
tidal::Options options{"cuda", 0, 6};
options.max_draft_tokens = 3;
options.mtp = true;                          // model's own MTP block, or:
options.mtp_model = "Qwen3.8-27B-MTP.gguf";  // sidecar with only the MTP tensors
tidal::Engine engine(model, options);
engine.prefill(prompt, false);
int32_t token = engine.sample();
for (;;) {
    auto emitted = engine.verify(token, engine.draft(token, 3));
    // emitted: accepted drafts, then one sampled token that is not yet consumed
    token = emitted.back();
}
```

## Acceptance

Verification samples the batch's positions in order with the decode sampler
(temperature, top-k/p, penalties and the same random stream). A draft is
accepted while it equals the sample at its position. The first sample that is
not an accepted draft ends the call and becomes the next `token`. For a
deterministic drafter such as greedy MTP, the emitted sequence has the same
distribution as `decode` + `sample`. With greedy sampling it is the same
sequence, up to rounding: the batch's matrix products round differently from
single-token decode, so a near-tie between the two highest logits may resolve
differently.

## Rollback

A verification batch advances every layer over all `1 + drafts` positions.
Rejected positions are undone after sampling:

- CPU KV rows are truncated.
- GDN layers run `gated_delta_net` with one state snapshot per position (slot
  `s` is the state `s` tokens before the last). The batch keeps the snapshots
  and the convolution history, so the state of any accepted length is
  restored by copies.
- QSA keeps the batch's raw index rows from the start of the first block and
  rebuilds the ring of the accepted length. Index-cache rows of blocks that a
  rejected position completed stay masked until their block completes again.
- The last consumed position's logits and hidden row are installed.

PLE models are not supported.

## MTP

The Qwen3-Next/Qwen3.5 MTP block predicts token `i + 2` from the main model's
final-normed hidden state at position `i` and the embedding of token `i + 1`:
`eh_proj(concat(enorm(embed), hnorm(hidden)))`, one full-attention decoder
layer, `shared_head_norm`, then the main output head. Its KV is a persistent
device cache over all positions, attended with FlashAttention.

MTP KV rows are written when the next token of a position is known:

- in layer-major prefill, from the batch's final hidden rows;
- in hybrid batches and verification, inside the head segment (rows past the
  accepted length are dropped);
- after a single decode, for the previous position.

The last consumed position waits as a backlog row until `draft` receives its
next token. Recursive drafts feed the block its own output and greedy token;
their KV rows are speculative and dropped. Prompt-cache snapshots include the
MTP position and backlog row.

## MTP weights

A GGUF converted with its MTP block (`<arch>.nextn_predict_layers = 1`) is used
directly with `mtp = true`. Some conversions omit the block. A sidecar GGUF then
supplies it: `general.architecture = qwen35`, `tidal.mtp_sidecar = true`,
`qwen35.block_count` = model blocks + 1, and the tensors of block
`<model blocks>` under llama.cpp's names:

| HF name | GGUF name (`blk.N.`) |
|---|---|
| `mtp.fc` | `nextn.eh_proj` |
| `mtp.pre_fc_norm_embedding` / `_hidden` | `nextn.enorm` / `nextn.hnorm` |
| `mtp.norm` | `nextn.shared_head_norm` |
| `mtp.layers.0.input_layernorm` / `post_attention_layernorm` | `attn_norm` / `post_attention_norm` |
| `mtp.layers.0.self_attn.{q,k,v,o}_proj`, `{q,k}_norm` | `attn_{q,k,v,output}`, `attn_{q,k}_norm` |
| `mtp.layers.0.mlp.{gate,up,down}_proj` | `ffn_{gate,up,down}` |

RMSNorm gammas are stored as `1 + w` in F32, as llama.cpp's qwen35 converter
does. Matrices keep their checkpoint type.

`tidal-cli --draft-tokens N --mtp mtp|SIDECAR.gguf --logits 0` generates
speculatively.
