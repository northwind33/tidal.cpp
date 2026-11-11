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
`POST /slots/0?action=erase` drops the cached sequence. Responses report reuse
as `usage.prompt_tokens_details.cached_tokens` (OpenAI),
`usage.cache_read_input_tokens` (Anthropic) and `tokens_cached` / `timings.cache_n`
(llama.cpp). Anthropic `cache_control` markers are accepted and unnecessary.

**Cancellation.** Closing the connection stops the generation at the next
token, for streamed and non-streamed requests alike. A prompt that is being
prefilled finishes its prefill first.

**Sampling.** Each request sets its own parameters with `Engine::set_sampling`,
which reseeds the generator and rebuilds the penalty history from the consumed
sequence; the sampler backend (`--sampler`) is fixed at startup. See
[sampling](sampling.md) for the semantics. `top_p: 0` selects greedy decoding.

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
| `POST /v1/chat/completions` | `messages` (text parts only), `tools`, `tool_choice`, `stream`, `stream_options.include_usage`, `max_tokens` / `max_completion_tokens`, `stop`, `seed`, `temperature`, `top_p`, `presence_penalty`, `frequency_penalty`, `logprobs` / `top_logprobs`. Extensions: `top_k`, `min_p`, `repeat_penalty`, `repeat_last_n`, `cache_prompt`, the thinking fields above; responses carry `timings`. |
| `POST /v1/completions` | `prompt` (string, token ids, or a mix), `max_tokens`, `stop`, `echo`, `logprobs`, `stream`. |
| `GET /v1/models` | The loaded model, in a shape both OpenAI and Anthropic clients accept. |

`/chat/completions` and `/models` are aliases. Only `n = 1` is supported.

### Anthropic

| Endpoint | Notes |
|---|---|
| `POST /v1/messages` | `system`, `messages` with `text`, `thinking`, `tool_use` and `tool_result` blocks, `tools`, `tool_choice`, `thinking`, `output_config.effort`, `max_tokens`, `stop_sequences`, `temperature`, `top_p`, `top_k`, `stream`. Streams the standard event sequence (`message_start`, `content_block_*`, `message_delta`, `message_stop`, `ping`). |
| `POST /v1/messages/count_tokens` | Token count of the rendered prompt. |

`usage.input_tokens` counts evaluated tokens and `cache_read_input_tokens` the
reused ones. Thinking blocks have an empty `signature`. Server-side tools
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
| `GET /slots`, `POST /slots/0?action=erase` | Slot state; erase drops the prompt cache. |
| `GET /metrics` | Prometheus text with the `llamacpp:` metric names, plus `tidal:prompt_tokens_cached_total`. |
| `GET /health`, `/v1/health` | 200 `{"status": "ok"}` once the model is ready, 503 while loading. |

### Not supported

Embeddings, reranking, LoRA adapters, the OpenAI Responses API, audio and image
endpoints answer 501. Images and other non-text content parts, grammars and
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
  BPE on a generated GGUF vocabulary, native template rendering and the
  streaming output parser (the same result however the text is chunked).
- `server` (`tests/test_server.py`): starts `tidal-server` on a `tidal-dummy`
  model with the CPU backend and checks every endpoint over HTTP: response
  shapes of the three API families, streamed against non-streamed output,
  seeds, token probabilities, stop strings, prompt-cache accounting, the
  context limit, keep-alive and chunked transfer, queued concurrent requests,
  client disconnects, API keys, command-line defaults, shutdown and startup
  failures.
