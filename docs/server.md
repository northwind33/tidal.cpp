# HTTP server

`tidal-server` keeps one model loaded and serves it over HTTP. One process
answers three families of clients:

- **OpenAI**: `/v1/chat/completions`, `/v1/completions`, `/v1/models`
- **Anthropic**: `/v1/messages`, `/v1/messages/count_tokens`
- **llama.cpp server**: `/completion`, `/infill`, `/tokenize`, `/detokenize`,
  `/apply-template`, `/props`, `/slots`, `/metrics`, `/health`

It also serves a single-page test client at `/`. The server has no dependencies
beyond the engine: HTTP, JSON, the tokenizer and the chat format are implemented
in `server/`. It builds on POSIX systems (`-DTIDAL_SERVER=OFF` leaves it out).

```sh
build/bin/tidal-server --model models/qwen35/Qwen3.5-0.8B-Q8_0.gguf --context 8192
# tidal-server: listening on http://127.0.0.1:8080, loading ...
# tidal-server: model ready (backend CUDA0, context 8192, chat format qwen35)

curl http://127.0.0.1:8080/v1/chat/completions -d '{
  "messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 64}'
```

The port opens before the model has loaded; until then `/health` answers 503
and other endpoints answer 503 with `Loading model`. If loading fails the
process exits with status 1.

## Options

```text
tidal-server --model MODEL.gguf [--host 127.0.0.1] [--port 8080] [--alias NAME] [--api-key KEY]
  [--backend cuda|cpu] [--device 0] [--threads 4] [--context N]
  [--draft-tokens 0] [--mtp mtp|SIDECAR.gguf] [--thinking auto|on|off]
  [--chat-template auto|chatml|qwen35|qwen38] [--n-predict -1] [--webui 1]
  [--sampler device|cpu] [--temp T] [--top-k K] [--top-p P] [--min-p P] [--repeat-penalty R]
  [--frequency-penalty F] [--presence-penalty F] [--penalty-last-n N] [--seed S]
  [--cache-min-tokens 1024] [--cache-snapshots 16]
  [--cache-ram SIZE] [--cache-disk SIZE --cache-dir DIR] [--cache-chunk-tokens 256] [--cache-branch-tokens 512]
```

The KV placement, precision and prefill options of `tidal-cli` are accepted too
(`tidal-server --help` lists them).

- `--context` bounds a sequence: prompt plus reply. It defaults to the model's
  capacity, and CPU KV memory is reserved for the whole context, so set it to
  what the deployment needs.
- `--host` defaults to the loopback interface. Use `--host 0.0.0.0` (or `::`)
  to accept remote clients, preferably with `--api-key`. With a key, every
  endpoint except `/health`, `/v1/models` and the test page requires
  `Authorization: Bearer KEY` or `x-api-key: KEY`. There is no TLS; put a
  reverse proxy in front for that.
- Sampling flags set the defaults of requests that carry no value. A flag that
  is not given falls back to the model's `general.sampling.*` GGUF metadata and
  then to temperature 0.8, top-k 40, top-p 0.95, min-p 0.05. Without `--seed`,
  a request without a seed draws a fresh one.
- `--n-predict` is the default reply limit; -1 runs until the context is full.
- `--draft-tokens N --mtp ...` generates with
  [speculative decoding](speculative.md). Requests that ask for token
  probabilities use plain decoding.
- `--alias` is the model name reported by `/v1/models`; it defaults to the file
  name. Responses echo the `model` a request names.
- `--cache-min-tokens` and `--cache-snapshots` configure
  [prompt caching](#prompt-caching).
- `--cache-ram`, `--cache-disk` and `--cache-dir` enable the
  [cache store](#cache-store), which lets conversations take turns on the
  engine without being evaluated again.

## Execution model

The engine holds a single sequence, so the server has **one slot**. Requests
queue and run one at a time in arrival order; `/slots` and `/metrics` show the
queue. Tokenization, template rendering and validation happen before queuing,
so an invalid request never waits.

**Prompt cache.** Every request goes through `Engine::prefill_cached`: the
prefix a prompt shares with the previous sequence is not evaluated again. A
follow-up turn that extends the previous prompt and reply reuses all of it;
otherwise the engine rolls back to the latest snapshot inside the common prefix
(one is taken at the end of each prompt, `--prompt-cache-checkpoints` bounds
them). `"cache_prompt": false` evaluates from scratch, and
`POST /slots/0?action=erase` drops the cached sequence. The llama.cpp
endpoints report the reuse as `tokens_cached` / `timings.cache_n`; the OpenAI and
Anthropic endpoints report cache usage as their APIs define it
([prompt caching](#prompt-caching)), and also carry `timings.cache_n`
where the response has `timings`.

**Cache store.** The engine holds one sequence, so on its own a second
conversation overwrites the first and the first is evaluated again when it
returns. With `--cache-ram` and/or `--cache-disk`, the state a request
overwrites is kept in host memory and on disk, and a prompt that returns to it
resumes from there ([cache store](#cache-store)).

**Cancellation.** Closing the connection stops the generation at the next
token, for streamed and non-streamed requests alike. A prompt that is being
prefilled finishes its prefill first.

**Sampling.** Each request sets its own parameters with `Engine::set_sampling`,
which reseeds the generator and rebuilds the penalty history from the consumed
sequence; the sampler backend (`--sampler`) is fixed at startup. See
[sampling](sampling.md) for the semantics. `top_p: 0` selects greedy decoding.

## Prompt caching

The engine reuses whatever prefix it can restore (see above); that is a matter
of speed only. What a response *reports* as cache reads and writes follows the
published behavior of each API, so that clients tuning their cache placement
see the numbers they would see from the hosted service:

- **Entries.** A request writes cache entries at its write breakpoints: an
  entry is a prompt prefix (its token ids and image contents) with a lifetime.
  A request reads the longest prefix that ends at one of its lookup boundaries
  and has a live entry. Reads never exceed what the engine actually reused: an
  entry whose state the engine can no longer restore is a miss, as an evicted
  entry would be. Lifetimes count from the start of the request.
- **Snapshots.** The prefill stops at every write breakpoint to take an engine
  snapshot there, and snapshots of live entries stay pinned (up to
  `--cache-snapshots`, most recently used first) outside the
  `--prompt-cache-checkpoints` limit. A later request that diverges after a
  breakpoint, such as a different question after a shared system prompt or
  another conversation, therefore restores it instead of starting over. A
  snapshot holds the recurrent state that KV truncation cannot rewind (about
  150 MB of host memory for the 27B Qwen3.8 model, 20 MB for the 0.8B one), so
  size `--cache-snapshots` to the memory available; 0 disables pinning. The
  engine holds one sequence, so pinned snapshots on a branch that another
  request rewinds are lost; their entries then miss.
- **Minimum.** A breakpoint caches only a prefix of at least
  `--cache-min-tokens` tokens (default 1024, the OpenAI minimum; Anthropic's
  is model-dependent). Shorter prefixes are silently not cached.
- **Blocks to tokens.** Breakpoints sit at the end of a content block. The
  template's rendered text is mapped back to tokens: a breakpoint covers the
  tokens that lie entirely before the block's end. The native templates render
  tools, then the system prompt, then the messages, as both APIs order them.

### OpenAI (`/v1/chat/completions`)

GPT-5.6 semantics:

| Field | Behavior |
|---|---|
| `prompt_cache_options.mode` | `implicit` (default): a breakpoint at the end of the latest eligible message (a user message, the last of a group of tool messages, or the last of the leading developer/system messages), plus up to the latest three explicit breakpoints. `explicit`: only explicit breakpoints, up to the latest four; without any, nothing is cached. |
| `prompt_cache_options.ttl` | `"30m"`, the only value Chat Completions accepts. |
| `prompt_cache_breakpoint` | On any content part: `{"mode": "explicit", "ttl": "30m" \| "12h"}`; the ttl defaults to the request's. |
| `prompt_cache_key` | Separates cache accounting: entries are only read by requests with the same key. |
| `prompt_cache_retention` | `in_memory` or `24h`; validated and otherwise ignored (it governs older models). |

Lookups consider the first 2 and latest 50 explicit breakpoints and, in
implicit mode, the implicit breakpoint, up to 20 earlier eligible message
endings and the end of the leading developer messages. Reads renew an entry
for 30 minutes. Each token is billed once, at the longest lifetime among the
written breakpoints that cover it; tokens whose existing entry is shorter-lived
than that are written again (an upgrade from 30-minute to 12-hour coverage)
rather than read. `usage.prompt_tokens_details` reports `cached_tokens` (reads)
and `cache_write_tokens`. `/v1/completions` has no messages and reports the
engine's reuse as `cached_tokens`.

### Anthropic (`/v1/messages`)

| Field | Behavior |
|---|---|
| `cache_control` on a block | `{"type": "ephemeral", "ttl": "5m" \| "1h"}` on tools, system text blocks and message `text`, `image`, `tool_use` and `tool_result` blocks (and blocks inside a `tool_result`). Not allowed on thinking blocks or empty text blocks. |
| top-level `cache_control` | Automatic caching: a breakpoint on the last block that can take one. An explicit marker there with the same ttl makes it a no-op; a different ttl is an error. |
| limits | At most 4 breakpoints, the automatic one included; a `1h` breakpoint may not follow a `5m` one. |
| `max_tokens: 0` | Pre-warming: the prompt is evaluated and cached, the reply has no content and `stop_reason` `max_tokens`. Not allowed with `stream`, `thinking.type: "enabled"` or a forced `tool_choice`. |

Each breakpoint writes an entry and looks for an earlier one over at most 20
block positions before it (a run of `tool_use` or of `tool_result` blocks is one
position). A read renews the entry's own lifetime. `usage` reports
`cache_read_input_tokens` (up to the hit), `cache_creation_input_tokens` (from
the hit to the last breakpoint, split by lifetime in
`cache_creation.ephemeral_5m_input_tokens` / `ephemeral_1h_input_tokens`) and
`input_tokens` (the rest). **A request without any `cache_control` keeps the
implicit behavior**: the prefix the engine reused is reported as
`cache_read_input_tokens` and nothing as written.

`POST /slots/0?action=erase` drops the cache entries with the sequence, and
`/metrics` adds `tidal:cache_read_tokens_total`, `tidal:cache_write_tokens_total`,
`tidal:cache_entries`, `tidal:cache_snapshots` and `tidal:cache_snapshot_bytes`.

## Cache store

```sh
tidal-server --model MODEL.gguf --context 65536 \
  --cache-ram 8G --cache-disk 100G --cache-dir /var/cache/tidal
```

Agents rarely take turns politely: session A asks, session B asks, A comes
back. The engine has one sequence, so without a store B overwrites A's KV rows
and A's whole conversation is prefilled again on its return. The cache store
keeps what would be lost, in as much host memory and disk space as the flags
allow, and brings it back:

- **What is stored.** KV rows, cut into chunks of `--cache-chunk-tokens`
  positions, and state snapshots (the recurrent state that KV rows alone cannot
  rebuild, with the rows since the last chunk boundary). Both are addressed by
  the tokens and image contents of the prefix they belong to, so conversations
  that share a system prompt and tool definitions share those chunks, and a
  conversation that grows only adds chunks.
- **When.** After each reply, while the engine is idle, the server copies the
  prompt's state into the store in the background; a request that arrives
  meanwhile interrupts the copy after the blob in progress. A request that
  replaces the sequence first saves whatever is not stored yet, up to the end
  of the last reply, so nothing is lost by switching quickly. `SIGTERM`/`SIGINT`
  save the current sequence before exiting.
- **Resuming.** Every prompt is matched against the store. If a stored
  snapshot lies deeper in the prompt than what the engine can reuse on its own,
  the rows beyond the prefix that the engine still holds are loaded and the
  snapshot is restored; only the rest of the prompt is evaluated. A
  conversation that comes back with its last reply unchanged resumes after that
  reply; one whose reply is rendered differently (thinking removed, for
  example) resumes at the end of its previous prompt.
- **Parting points.** A prompt that leaves another conversation
  `--cache-branch-tokens` or more past the state it resumes from takes a
  snapshot where the two part, typically the end of a shared system prompt.
  Later conversations with that prefix start from there. Zero disables it.
- **Budgets.** `--cache-ram` bounds the host memory and `--cache-disk` the
  files under `--cache-dir`; sizes take `K`, `M`, `G` or `T` and are MiB
  without a suffix. New state enters memory. When memory fills up, the least
  recently used state is written to disk in the background and dropped from
  memory; state read back from disk stays in memory while there is room. When
  neither tier has room, the least recently used conversation goes, deepest
  state first, so that nothing is kept that could no longer be restored.
  Snapshots that a conversation has moved past are the first to go and are
  never written to disk. Either flag alone works: memory only, or disk only.
- **Sizing.** A snapshot is as large as a pinned one (about 150 MB for the 27B
  Qwen3.8 model); the KV rows are what the engine's own KV cache holds per
  token. `--cache-ram` smaller than one snapshot only works together with a
  disk budget. `GET /slots` shows the usage.
- **Restarts.** Files are tied to the model file and to the options that shape
  the state (backend, KV placement and precision, context, chunk size, MTP,
  vision model). A server started with the same ones adopts the files and
  resumes conversations from them; with other ones it uses a separate
  directory below `--cache-dir`. Files are checksummed: a damaged file is
  dropped and its conversation is evaluated instead.

A restore brings back exactly the saved bytes, so a conversation continues as
if it had never left. Chunks of a shared prefix hold the rows of whichever
conversation stored them first; on CUDA those may differ from another
conversation's own rows in the last bits, as prefill and decode do.

`"cache_prompt": false` evaluates the whole prompt but still saves the sequence
it replaces. `POST /slots/0?action=erase` empties the store, files included.
`timings.cache_store_n` is the part of `cache_n` that came from the store
(beyond what the engine itself still held); `timings.cache_store_ms` is the
time loading it took and `timings.cache_save_ms` the time spent saving the
sequence the request replaced, both part of `prompt_ms`. `GET /slots` and
`GET /props` carry a `cache_store` object (budgets, bytes in memory and on
disk, snapshots, stored token positions, restores, time spent restoring and
saving, disk traffic, the latest disk error), and `/metrics` adds
`tidal:cache_store_ram_bytes`, `tidal:cache_store_disk_bytes`,
`tidal:cache_store_snapshots`, `tidal:cache_store_tokens`,
`tidal:cache_store_restores_total`, `tidal:cache_store_restored_tokens_total`,
`tidal:cache_store_restore_seconds_total`, `tidal:cache_store_save_seconds_total`,
`tidal:cache_store_written_bytes_total`, `tidal:cache_store_read_bytes_total`
and `tidal:cache_store_dropped_total`. After a disk failure the store keeps
serving what it has and stops writing.

The API cache accounting above is unchanged: a response reports as read only
what the engine reused, and with the store that now includes entries whose
state another conversation had overwritten.

## Chat format

The Jinja template in the GGUF is **not interpreted**. The server recognizes
the Qwen3.5 and Qwen3.8 templates and renders them natively; rendering was
checked against llama.cpp's Jinja engine for both templates (system prompts,
multi-turn reasoning, tool calls and tool responses, thinking on and off).
Any other template, or none, falls back to plain ChatML without tools or
thinking. `--chat-template` overrides the detection, `/props` reports the
result as `chat_format`, and `/apply-template` shows the rendered prompt.

| | `qwen35` | `qwen38` |
|---|---|---|
| Thinking when the request does not say | off | on |
| `reasoning_effort` (`low`, `medium`, `xhigh`) | ignored | adds the template's system instruction |
| Reasoning of earlier turns | dropped before the last user query | kept (`preserve_thinking: false` drops it) |

Template behavior carried over as is: the system message must come first
(several leading system messages are joined), a conversation needs a user
query, tools are listed in the system turn, and tool calls use the
`<tool_call><function=NAME><parameter=KEY>` form.

**Thinking.** Generated `<think>` sections are returned separately:
`reasoning_content` (OpenAI message and stream delta) or a `thinking` block
(Anthropic). Whether a reply thinks is decided, in increasing priority, by
`--thinking`, then `reasoning_effort` (`none` disables), `thinking.type`,
`enable_thinking`, and `chat_template_kwargs.enable_thinking` in the request.
The Messages API thinks only when the request has a `thinking` field, as the
Anthropic API does. `reasoning_format: "none"` leaves the section in the content,
tags included.

**Tools.** Tool calls in the output are parsed into `tool_calls` (OpenAI) or
`tool_use` blocks (Anthropic). Argument values are typed with the tool's JSON
schema: `string` parameters stay literal, the others are parsed as JSON. A
stream delivers each call whole once it is complete. `tool_choice` `required` /
`any`, or a named function, is enforced by starting the reply inside a tool
call (thinking is then off); `none` hides the tools. Output is otherwise
unconstrained: `response_format` other than `text`, `grammar` and `json_schema`
are rejected with 400 rather than ignored.

**Assistant prefill.** When the last message is an assistant message without
tool calls, the reply continues that text instead of starting a new turn.

## Tokenizer

The tokenizer is read from the GGUF: byte-level BPE (`tokenizer.ggml.model =
gpt2`) with the `qwen2` or `qwen35` pre-tokenizer. Token ids were compared with
llama.cpp on multilingual text, source code, whitespace runs, combining marks
and special tokens. Like llama.cpp, it does not apply Unicode NFC normalization;
text that is not already NFC (decomposed Hangul or accents) tokenizes
differently from the Hugging Face tokenizer. Other tokenizer types are rejected
at startup.

A GGUF without a tokenizer, such as a `tidal-dummy` model, gets a byte
tokenizer (one token per UTF-8 byte) and ChatML. That is what the tests use.

Generated bytes are emitted as complete UTF-8 characters; an ill-formed
sequence becomes U+FFFD. Control tokens are not rendered and end-of-turn tokens
end the reply.

## Endpoints

### OpenAI

| Endpoint | Notes |
|---|---|
| `POST /v1/chat/completions` | `messages` (text parts only), `tools`, `tool_choice`, `stream`, `stream_options.include_usage`, `max_tokens` / `max_completion_tokens`, `stop`, `seed`, `temperature`, `top_p`, `presence_penalty`, `frequency_penalty`, `logprobs` / `top_logprobs`. `prompt_cache_options`, `prompt_cache_breakpoint`, `prompt_cache_key`, `prompt_cache_retention`. Extensions: `top_k`, `min_p`, `repeat_penalty`, `repeat_last_n`, `cache_prompt`, the thinking fields above; responses carry `timings`. |
| `POST /v1/completions` | `prompt` (string, token ids, or a mix), `max_tokens`, `stop`, `echo`, `logprobs`, `stream`. |
| `GET /v1/models` | The loaded model, in a shape both OpenAI and Anthropic clients accept. |

`/chat/completions` and `/models` are aliases. Only `n = 1` is supported.

### Anthropic

| Endpoint | Notes |
|---|---|
| `POST /v1/messages` | `system`, `messages` with `text`, `thinking`, `tool_use` and `tool_result` blocks, `tools`, `tool_choice`, `thinking`, `output_config.effort`, `max_tokens` (0 pre-warms the cache), `stop_sequences`, `temperature`, `top_p`, `top_k`, `stream`, `cache_control` (on blocks and top-level). Streams the standard event sequence (`message_start`, `content_block_*`, `message_delta`, `message_stop`, `ping`). |
| `POST /v1/messages/count_tokens` | Token count of the rendered prompt. |

`usage` follows [prompt caching](#prompt-caching). Thinking blocks have an empty `signature`. Server-side tools
(entries without `input_schema`) are skipped. Errors use the Anthropic envelope
`{"type": "error", "error": {...}}`.

### llama.cpp

| Endpoint | Notes |
|---|---|
| `POST /completion` | `prompt` (string, token ids, or a mix; special tokens are parsed), `n_predict`, sampling fields, `stop`, `stream`, `cache_prompt`, `n_probs`, `return_tokens`, `special`. The response has `content`, `tokens`, `stop_type` (`eos`, `limit`, `word`), `stopping_word`, `tokens_predicted`, `tokens_evaluated`, `tokens_cached`, `generation_settings` and `timings`. `/completions` is an alias. |
| `POST /infill` | `input_prefix`, `input_suffix`; needs the model's `<|fim_*|>` tokens. |
| `POST /tokenize` | `content`, `parse_special` (default true), `add_special`, `with_pieces`. |
| `POST /detokenize` | `tokens`. |
| `POST /apply-template` | The chat request body; returns `{"prompt": ...}` without generating. |
| `GET /props` | Context size, default sampling, chat template and format, tokenizer kind. |
| `GET /slots`, `POST /slots/0?action=erase` | Slot state; erase drops the prompt cache and its cache entries. |
| `GET /metrics` | Prometheus text with the `llamacpp:` metric names, plus `tidal:prompt_tokens_cached_total`, `tidal:images_encoded_total` and the `tidal:cache_*` metrics of [prompt caching](#prompt-caching). |
| `GET /placement`, `/v1/placement` | Every weight, KV cache and state tensor with its memory and compute device ([vision guide](vision.md#tensor-placement-api)). |
| `GET /health`, `/v1/health` | 200 `{"status": "ok"}` once the model is ready, 503 while loading. |

### Not supported

Embeddings, reranking, LoRA adapters, the OpenAI Responses API, audio and image
generation endpoints answer 501. Image inputs need `--vision-model` (see
[vision](vision.md)); remote image URLs, video and audio parts, grammars and
JSON-schema constrained output, `n > 1`, slot save/restore and parallel slots
are rejected with an error message that names the limitation.

## Clients

```sh
# OpenAI SDK
python - <<'EOF'
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
print(client.chat.completions.create(model="tidal", messages=[{"role": "user", "content": "Hi"}]).choices[0].message.content)
EOF

# Anthropic SDK
python - <<'EOF'
from anthropic import Anthropic
client = Anthropic(base_url="http://127.0.0.1:8080", api_key="unused")
print(client.messages.create(model="tidal", max_tokens=256, messages=[{"role": "user", "content": "Hi"}]).content[0].text)
EOF

# Raw completion, streamed
curl -N http://127.0.0.1:8080/completion -d '{"prompt": "The capital of France is", "n_predict": 16, "stream": true}'
```

### Test page

`http://127.0.0.1:8080/` exercises the server from a browser, one tab per
feature: chat over either the OpenAI or the Anthropic API (streaming, thinking,
tools with a manual tool-result round trip, sampling fields, usage and speed),
raw completion and infill, tokenize / detokenize / count tokens, template
rendering, and the status endpoints. Each chat shows the exact request and
response. `--webui 0` disables the page.

## Verification

```sh
ctest --test-dir build -R server --output-on-failure
```

- `server-text` (`tests/server.cpp`): JSON, UTF-8 handling, pre-tokenization,
  BPE on a generated GGUF vocabulary, native template rendering and its layout
  (where each message, tool and call lands in the text), the streaming output
  parser (the same result however the text is chunked), and the cache
  accounting: lifetimes, renewal, lookback, namespaces and the OpenAI billing
  table for overlapping writes and lifetime upgrades.
- `server` (`tests/test_server.py`): starts `tidal-server` on a `tidal-dummy`
  model with the CPU backend and checks every endpoint over HTTP: response
  shapes of the three API families, streamed against non-streamed output,
  seeds, token probabilities, stop strings, prompt-cache accounting, OpenAI and
  Anthropic prompt caching with exact token counts (the byte tokenizer makes
  positions byte offsets), the
  context limit, keep-alive and chunked transfer, queued concurrent requests,
  client disconnects, API keys, command-line defaults, shutdown and startup
  failures.
