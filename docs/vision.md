# Vision input

Images enter the decoder through the vision tower released with Qwen3.5,
Qwen3.8 and Qwen3.8-Flash-Next (`qwen4exp`): a Qwen3-VL ViT with a 2x2 patch
merger. Its weights load from an **mmproj GGUF** (`general.architecture = clip`,
`clip.projector_type = qwen3vl_merger`), the format llama.cpp's converter writes
for these models; the tower's output width must equal the text model's hidden
size, so each mmproj belongs to one text model.

| Text model | Public mmproj (examples) | Tower |
|---|---|---|
| Qwen3.5-0.8B (`qwen35`) | `unsloth/Qwen3.5-0.8B-GGUF`: `mmproj-F32.gguf`, `mmproj-BF16.gguf` | 12 layers, width 768, output 1024 |
| Qwen3.8-27B (`qwen35`) | `ggml-org/Qwen3.8-27B-GGUF`: `mmproj-Qwen3.8-27B-BF16.gguf`, `-Q8_0` | 27 layers, width 1152, output 5120 |
| Qwen3.8-Flash-Next (`qwen4exp`) | `ggml-org/Qwen3.8-Flash-Next-GGUF`: `mmproj-Qwen3.8-Flash-Next-Q8_0.gguf` | 27 layers, width 1152, output 2560 |

The synthetic `tidal-qwen4exp-*` text models are random, so a released tower is
only meaningful with a released checkpoint; for tests, `tidal-dummy --vision 1`
writes a random tower in the same format, and `--image-token ID` records the
placeholder token of a synthetic text model (`tidal.image_token`). A synthetic
text model with `--hidden 2560` accepts the released Flash-Next tower.

```bash
tidal-server --model models/qwen38/Qwen3.8-27B-Q4_K_M.gguf \
  --vision-model models/qwen38/mmproj-Qwen3.8-27B-BF16.gguf --context 32768
tidal-vision --model M.gguf --vision-model MMPROJ.gguf --images photo.jpg \
  --tokens 248045,846,198,248053,IMG,248054,... --steps 64
```

## Weight placement

`--vision-placement` (`Options::vision_placement`) chooses where the tower's
weights live, and the encoder runs where its weights are:

- `gpu`: weights in VRAM, encoder graph on the CUDA device.
- `cpu`: weights in system memory, encoder graph on CPU threads (`--threads`);
  no VRAM is used by the tower, at the cost of slower image encoding.
- `auto` (default): `gpu` with `--backend cuda`, `cpu` with `--backend cpu`.

There is no silent fallback: an unsupported operation or a failed allocation is
an error. Matrices keep their stored type (F32, F16, BF16, Q8_0, ...) when the
device multiplies it natively and are widened to F32 otherwise; norms and biases
are F32. The learned position table stays in host memory because positions are
interpolated per image while preprocessing.

## Preprocessing

1. Decoding (`tidal::decode_image`, no external libraries): PNG (bit depths 1-16,
   gray, RGB, palette, alpha, `tRNS`, Adam7), JPEG (baseline and progressive
   Huffman, 4:4:4/4:2:2/4:2:0 with triangle upsampling, restart intervals, gray,
   Adobe CMYK/YCCK), BMP (8/24/32-bit) and binary PPM/PGM. Alpha is composited
   over white, as the Transformers processors do. WebP and GIF are rejected.
2. Smart resize: both sides become multiples of 32 (patch 16 x merge 2) near the
   original aspect ratio, with the token count (pixels / 1024) inside
   `[--image-min-tokens, --image-max-tokens]`, by default 8-4096 (the llama.cpp
   defaults; a file may declare `clip.vision.image_min_pixels`/`max_pixels`).
   Aspect ratios beyond 200:1 are rejected.
3. Pillow-compatible bicubic resampling (a = -0.5, antialiased when shrinking,
   22-bit fixed-point weights, horizontal pass then vertical pass on 8-bit data),
   then `(x / 255 - mean) / std` with the file's mean and std.

## Tower

A still image is one temporal patch: both halves of the Conv3D kernel see the
same frame, so the patch embedding is `W0·p + W1·p + b` over 16x16x3 patches.
Patches are ordered so that each 2x2 merge group is consecutive. Each patch adds
the learned position table bilinearly interpolated (aligned corners) to the
patch grid. Every block is pre-norm (LayerNorm): fused QKV with bias, 2D rotary
embedding (the first quarter of each head's frequencies follows the patch row,
the second its column; base 10000), full non-causal attention, output projection,
then an MLP with the tanh GELU. The merger applies a final LayerNorm per patch,
concatenates each 2x2 group, and projects with fc1, exact (erf) GELU and fc2 to
the text width. DeepStack layers are not implemented (all released towers here
have none). Attention uses `ggml_flash_attn_ext` with FP16 K/V and F32
accumulation when the device supports the head size, and an explicit
softmax(QK^T)V otherwise.

## Prompts with images

`Engine::prefill(tokens, images)` and `Engine::prefill_cached(tokens, images)`
take the prompt's token IDs together with `ImageSpan{offset, image}` entries.
Each image occupies `Engine::image_tokens(width, height)` placeholder tokens at
`offset`; they must equal `Config::image_token` when the model declares it
(`<|image_pad|>`, found in the Qwen vocabulary, or `tidal.image_token`). The
chat templates render an image as `<|vision_start|><|image_pad|><|vision_end|>`;
the single `<|image_pad|>` expands to the image's tokens.

- **Embeddings.** Image positions take the tower's rows instead of token
  embeddings in every prefill path: layer-major CUDA prefill, CPU-KV hybrid
  batches, persistent GPU KV and the sequential CPU reference. PLE still hashes
  the placeholder token at those positions, as the released checkpoints do.
- **Interleaved MRoPE.** Every sequence index has a rotary position (t, h, w):
  text uses t = h = w; image token j of a grid with W columns uses
  (s, s + j / W, s + j % W) where s is the image's start, and the text after it
  continues at s + max(rows, columns). Rotary pairs follow t, h or w by pair
  index modulo 3 within `rope.dimension_sections` (`[11, 11, 10]` for the
  released models; synthetic models default to the same proportions). With equal
  axes this is exactly NeoX RoPE, so text-only execution is unchanged. KV rows,
  causal masks and QSA blocks keep using sequence indices; a QSA block key
  rotates at the position of the block's first token, which may be an image token.
- **Prompt cache.** A consumed position matches only if both its token and its
  image content hash match, so two prompts that differ only in an image share
  only the prefix before it. Images inside the reused prefix are not re-encoded,
  and the eight most recent embeddings are kept for prompts that miss the cache.
- **Speculative decoding and MTP** continue after image prompts with text
  positions; the MTP block sees the placeholder token's embedding at image
  positions, which affects draft quality only.

## Tensor placement API

`Engine::tensor_placements()` lists every weight of the text model, its MTP block
and the vision tower, plus the KV caches and recurrent states, with their type,
shape, bytes, the buffer that holds them (`memory`, e.g. `CUDA0`, `CPU`) and the
device whose kernels read them (`compute`). The CPU KV cache of an attention
layer is one entry in host memory computed by CPU threads. Weight groups that
layer-major prefill evicts show `resident: false` while evicted.

`tidal-server` serves the list at `GET /placement` (also `/v1/placement`) with
per-(component, role, memory, compute) totals; `?component=text|mtp|vision`,
`?role=weight|state|kv_cache` and `?tensors=0` (summary only) narrow it.
`tidal-vision --placement 1` prints the same list.

## Server

With `--vision-model`, chat messages accept OpenAI `image_url` parts (and
`input_image`), Anthropic `image` blocks with a base64 or data-URI source, and
llama.cpp `/completion` prompts `{"prompt_string": "... <__media__> ...",
"multimodal_data": [base64, ...]}`. Images must be inline data: the server does
not fetch `http(s)` URLs. System messages cannot contain images. `/props`
reports `modalities.vision` and the tower; `/metrics` counts encoded images.
The test page attaches images to chat messages.

## Verification

`ctest -R vision` runs `tests/test_vision.py` (CPU sequential path as
`vision-cpu`; hybrid, layer-major, GPU-KV and PLE paths as `vision` on CUDA):

- exact decoding of PNG (every color type and depth, Adam7, alpha and `tRNS`),
  BMP and PPM/PGM; JPEG variants written by `tests/image_codecs.py` (baseline,
  progressive with successive approximation, restart markers, 4:2:0, gray)
  decode identically to each other and close to the source;
- the tower against `tests/vision_reference.py`, an independent NumPy
  implementation including Pillow's resampling arithmetic;
- prompts with two images through every prefill path against the NumPy text
  oracle extended with image rows and interleaved MRoPE, including greedy
  decode steps after the images;
- prompt-cache reuse without re-encoding, rejection of wrong placeholders,
  mismatched towers and non-mmproj files, and the HTTP API including
  `/placement`.

For released models, compare with llama.cpp's mtmd (`llama-server --mmproj`):
greedy replies to the same prompt and image, and tower outputs for the same
decoded pixels. llama.cpp decodes images with stb_image, whose JPEG IDCT and
upsampling round slightly differently; feed both the same PPM to isolate the
tower. The merger GELU here is exact (erf), as in Transformers.
