#!/usr/bin/env python3
"""End-to-end checks of tidal-server over HTTP with a small synthetic model.

The model has random weights and no tokenizer, so the server uses its byte
tokenizer and plain ChatML. That is enough to check the protocol: request
validation, the OpenAI, Anthropic and llama.cpp response shapes, streaming,
stop strings, prompt-cache accounting (OpenAI and Anthropic caching semantics),
queuing, authentication and shutdown.
Chat-template and tokenizer correctness are covered by tests/server.cpp.
"""
import argparse
import http.client
import json
import re
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path

CONTEXT = 192


class Server:
    def __init__(self, binary, *flags):
        self.log = tempfile.TemporaryFile(mode='w+')
        self.process = subprocess.Popen([str(binary), '--host', '127.0.0.1', '--port', '0', *map(str, flags)],
                                        stdout=subprocess.DEVNULL, stderr=self.log)
        self.port = None
        deadline = time.time() + 60
        while time.time() < deadline and self.port is None and self.process.poll() is None:
            match = re.search(r'listening on http://127\.0\.0\.1:(\d+)', self.output())
            if match:
                self.port = int(match.group(1))
            else:
                time.sleep(0.02)
        if self.port is None:
            return
        while time.time() < deadline and self.process.poll() is None:
            try:
                if self.request('GET', '/health')[0] == 200:
                    return
            except OSError:  # The server gave up between the two checks.
                pass
            time.sleep(0.02)

    def output(self):
        self.log.seek(0)
        return self.log.read()

    def connection(self):
        return http.client.HTTPConnection('127.0.0.1', self.port, timeout=60)

    def request(self, method, path, body=None, headers=None):
        connection = self.connection()
        payload = None if body is None else body if isinstance(body, bytes) else json.dumps(body).encode()
        connection.request(method, path, payload, {'Content-Type': 'application/json', **(headers or {})})
        response = connection.getresponse()
        text = response.read().decode()
        connection.close()
        try:
            return response.status, json.loads(text)
        except json.JSONDecodeError:
            return response.status, text

    def post(self, path, body, status=200, headers=None):
        code, data = self.request('POST', path, body, headers)
        assert code == status, (path, code, data)
        return data

    def get(self, path, status=200):
        code, data = self.request('GET', path)
        assert code == status, (path, code, data)
        return data

    def events(self, path, body):
        """Server-sent events of a streamed response as (event, data) pairs."""
        connection = self.connection()
        connection.request('POST', path, json.dumps(body).encode(), {'Content-Type': 'application/json'})
        response = connection.getresponse()
        assert response.status == 200, (path, response.status, response.read())
        assert response.getheader('Content-Type') == 'text/event-stream'
        result = []
        for block in response.read().decode().split('\n\n'):
            if not block:
                continue
            name, data = '', None
            for line in block.split('\n'):
                if line.startswith('event: '):
                    name = line[7:]
                elif line.startswith('data: '):
                    data = line[6:]
            assert data is not None, block
            result.append((name, data if data == '[DONE]' else json.loads(data)))
        connection.close()
        return result

    def stop(self):
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
        code = self.process.wait(timeout=30)
        self.log.close()
        return code


def llama_endpoints(server):
    assert server.get('/health') == {'status': 'ok'} and server.get('/v1/health') == {'status': 'ok'}
    props = server.get('/props')
    assert props['default_generation_settings']['n_ctx'] == CONTEXT and props['total_slots'] == 1
    assert props['tokenizer'] == 'bytes' and props['chat_format'] == 'chatml'
    assert props['default_generation_settings']['params']['temperature'] == 0.8

    text = 'héllo ✓'
    raw = list(text.encode())
    assert server.post('/tokenize', {'content': text}) == {'tokens': raw}
    assert server.post('/tokenize', {'content': ['a', 98, 'c']}) == {'tokens': [97, 98, 99]}
    pieces = server.post('/tokenize', {'content': 'aé', 'with_pieces': True})['tokens']
    assert pieces == [{'id': 97, 'piece': 'a'}, {'id': 0xC3, 'piece': [0xC3]}, {'id': 0xA9, 'piece': [0xA9]}]
    assert server.post('/detokenize', {'tokens': raw}) == {'content': text}
    assert server.post('/detokenize', {'tokens': [0xC3]}) == {'content': '�'}
    server.post('/detokenize', {'tokens': [256]}, 400)
    server.post('/tokenize', {'content': 5.5}, 400)

    prompt = 'The quick brown fox'
    body = {'prompt': prompt, 'n_predict': 12, 'temperature': 0, 'return_tokens': True}
    first = server.post('/completion', body)
    assert first['stop'] is True and first['stop_type'] == 'limit' and first['stopped_limit'] is True
    assert first['tokens_predicted'] == 12 and len(first['tokens']) == 12 and first['tokens_evaluated'] == len(prompt)
    assert first['tokens_cached'] == 0 and first['timings']['predicted_n'] == 12 and first['timings']['prompt_n'] == len(prompt)
    assert first['content'] == bytes(first['tokens']).decode(errors='replace')
    # The same prompt again is served from the prompt cache and repeats the greedy output.
    second = server.post('/completion', body)
    assert second['tokens'] == first['tokens'] and second['tokens_cached'] == len(prompt) and second['timings']['prompt_n'] == 0
    assert server.post('/completion', {**body, 'cache_prompt': False})['tokens_cached'] == 0
    # A longer prompt reuses the snapshot at the end of the earlier one.
    longer = server.post('/completion', {**body, 'prompt': prompt + ' jumps'})
    assert longer['tokens_cached'] == len(prompt), longer['tokens_cached']
    assert server.post('/completion', {**body, 'prompt': list(prompt.encode())})['tokens'] == first['tokens']
    assert server.post('/completions', body)['tokens'] == first['tokens']

    events = server.events('/completion', {**body, 'stream': True})
    assert all(name == '' for name, _ in events)
    chunks = [data for _, data in events]
    assert [c['stop'] for c in chunks] == [False] * (len(chunks) - 1) + [True]
    assert ''.join(c['content'] for c in chunks) == first['content']
    assert [t for c in chunks for t in c['tokens']] == first['tokens']
    assert chunks[-1]['stop_type'] == 'limit' and chunks[-1]['tokens_predicted'] == 12

    # Sampling: a seed reproduces the draw; logits back the reported probabilities.
    sampled = {'prompt': prompt, 'n_predict': 16, 'temperature': 1.5, 'top_k': 0, 'top_p': 1, 'min_p': 0, 'return_tokens': True}
    draws = [server.post('/completion', {**sampled, 'seed': seed})['tokens'] for seed in (1, 1, 2, 3, 4)]
    assert draws[0] == draws[1] and any(draw != draws[0] for draw in draws[2:])
    probabilities = server.post('/completion', {**body, 'n_probs': 3})['completion_probabilities']
    assert [p['id'] for p in probabilities] == first['tokens']
    for entry in probabilities:
        top = entry['top_logprobs']
        assert len(top) == 3 and top[0]['id'] == entry['id'] and entry['logprob'] <= 0
        assert top[0]['logprob'] >= top[1]['logprob'] >= top[2]['logprob']
        assert bytes(entry['bytes']) == bytes([entry['id']])

    # A stop string ends the text before it, also when it spans several tokens.
    text = first['content']
    middle = len(text) // 2
    word = text[middle:middle + 2]
    stopped = server.post('/completion', {**body, 'stop': ['never-matches', word]})
    assert stopped['content'] == text[:text.find(word)] and stopped['stop_type'] == 'word' and stopped['stopping_word'] == word
    streamed = [data for _, data in server.events('/completion', {**body, 'stop': [word], 'stream': True})]
    assert ''.join(c['content'] for c in streamed) == stopped['content'] and streamed[-1]['stop_type'] == 'word'

    # Without a limit the reply runs until the context is full.
    filled = server.post('/completion', {'prompt': prompt, 'temperature': 0})
    assert filled['tokens_predicted'] == CONTEXT - len(prompt) and filled['stop_type'] == 'limit'
    assert server.post('/completion', {'prompt': prompt, 'n_predict': 0})['content'] == ''
    error = server.post('/completion', {'prompt': 'x' * CONTEXT}, 400)['error']
    assert error['type'] == 'exceed_context_size_error' and error['code'] == 400
    server.post('/completion', {}, 400)
    server.post('/completion', {'prompt': 'x', 'temperature': -1}, 400)
    server.post('/completion', {'prompt': 'x', 'grammar': 'root ::= "a"'}, 400)
    server.post('/infill', {'input_prefix': 'a', 'input_suffix': 'b'}, 501)

    slots = server.get('/slots')
    assert len(slots) == 1 and slots[0]['id'] == 0 and slots[0]['n_ctx'] == CONTEXT and slots[0]['is_processing'] is False
    assert server.post('/slots/0?action=erase', {}) == {'id_slot': 0, 'n_erased': 0}
    assert server.post('/completion', body)['tokens_cached'] == 0
    server.post('/slots/0?action=save', {}, 501)
    server.post('/slots/1?action=erase', {}, 404)
    metrics = server.get('/metrics')
    values = dict(line.split(' ') for line in metrics.splitlines() if not line.startswith('#'))
    assert float(values['llamacpp:tokens_predicted_total']) > 100 and float(values['llamacpp:prompt_tokens_total']) > 0
    assert float(values['tidal:prompt_tokens_cached_total']) >= len(prompt) and values['llamacpp:requests_processing'] == '0'
    return first


def openai_endpoints(server, reference):
    models = server.get('/v1/models')
    assert models['object'] == 'list' and models['data'][0]['id'] == 'unit-model' and models['data'][0]['object'] == 'model'
    assert server.get('/models') == models

    messages = [{'role': 'system', 'content': 'Be brief.'}, {'role': 'user', 'content': [{'type': 'text', 'text': 'Hi'}]}]
    rendered = '<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n'
    assert server.post('/apply-template', {'messages': messages}) == {'prompt': rendered}
    body = {'model': 'custom-name', 'messages': messages, 'max_tokens': 10, 'temperature': 0}
    reply = server.post('/v1/chat/completions', body)
    assert reply['object'] == 'chat.completion' and reply['model'] == 'custom-name' and reply['id'].startswith('chatcmpl-')
    choice = reply['choices'][0]
    assert choice['index'] == 0 and choice['finish_reason'] == 'length' and choice['message']['role'] == 'assistant'
    # The prompt is shorter than the 1024-token minimum, so nothing is cached in the OpenAI sense.
    assert reply['usage'] == {'prompt_tokens': len(rendered), 'completion_tokens': 10, 'total_tokens': len(rendered) + 10,
                              'prompt_tokens_details': {'cached_tokens': 0, 'cache_write_tokens': 0}}
    assert server.post('/chat/completions', body)['choices'][0]['message'] == choice['message']

    events = server.events('/v1/chat/completions', {**body, 'stream': True, 'stream_options': {'include_usage': True}})
    assert events[-1][1] == '[DONE]'
    chunks = [data for _, data in events[:-1]]
    assert all(c['object'] == 'chat.completion.chunk' and c['id'] == chunks[0]['id'] for c in chunks)
    assert chunks[0]['choices'][0]['delta'] == {'role': 'assistant', 'content': None}
    assert ''.join(c['choices'][0]['delta'].get('content') or '' for c in chunks if c['choices']) == choice['message']['content']
    assert chunks[-2]['choices'][0]['finish_reason'] == 'length' and chunks[-2]['choices'][0]['delta'] == {}
    assert chunks[-1]['choices'] == [] and chunks[-1]['usage']['completion_tokens'] == 10
    assert chunks[-1]['usage']['prompt_tokens_details'] == {'cached_tokens': 0, 'cache_write_tokens': 0}
    assert chunks[-2]['timings']['cache_n'] == len(rendered)  # The engine still reused the whole prompt.

    scored = server.post('/v1/chat/completions', {**body, 'logprobs': True, 'top_logprobs': 2})['choices'][0]['logprobs']['content']
    assert len(scored) == 10 and all(len(entry['top_logprobs']) == 2 and entry['logprob'] <= 0 for entry in scored)
    # An assistant message at the end is continued rather than answered.
    resumed = server.post('/apply-template', {'messages': messages + [{'role': 'assistant', 'content': 'Sure'}]})['prompt']
    assert resumed == rendered + 'Sure'

    prompt = 'The quick brown fox'
    legacy = server.post('/v1/completions', {'prompt': prompt, 'max_tokens': 12, 'temperature': 0})
    assert legacy['object'] == 'text_completion' and legacy['choices'][0]['text'] == reference['content']
    assert legacy['choices'][0]['finish_reason'] == 'length' and legacy['usage']['completion_tokens'] == 12
    echoed = server.post('/v1/completions', {'prompt': prompt, 'max_tokens': 12, 'temperature': 0, 'echo': True, 'logprobs': 1})
    assert echoed['choices'][0]['text'] == prompt + reference['content'] and len(echoed['choices'][0]['logprobs']['tokens']) == 12
    events = server.events('/v1/completions', {'prompt': prompt, 'max_tokens': 12, 'temperature': 0, 'stream': True})
    assert events[-1][1] == '[DONE]' and ''.join(data['choices'][0]['text'] for _, data in events[:-1]) == reference['content']

    for bad in [{}, {'messages': []}, {'messages': [{'role': 'user'}], 'n': 2}, {'messages': 'hi'},
                {'messages': [{'role': 'user', 'content': 'x'}], 'response_format': {'type': 'json_object'}},
                {'messages': [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'x'}}]}]},
                {'messages': [{'role': 'user', 'content': 'x'}], 'max_tokens': 'many'},
                {'messages': [{'role': 'user', 'content': 'x'}, {'role': 'wizard', 'content': 'x'}]}]:
        error = server.post('/v1/chat/completions', bad, 400)['error']
        assert error['type'] == 'invalid_request_error' and error['code'] == 400 and error['message']
    assert server.post('/v1/chat/completions', b'{"messages": [', 400)['error']['type'] == 'invalid_request_error'
    assert server.post('/v1/embeddings', {'input': 'x'}, 501)['error']['type'] == 'not_supported_error'
    assert server.get('/v1/unknown', 404)['error']['type'] == 'not_found_error'
    assert server.get('/v1/chat/completions', 405)['error']['code'] == 405


def anthropic_endpoints(server):
    body = {'model': 'claude-test', 'max_tokens': 9, 'temperature': 0, 'system': [{'type': 'text', 'text': 'Be brief.'}],
            'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'Hi'}]}]}
    rendered = '<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n'
    assert server.post('/v1/messages/count_tokens', {k: v for k, v in body.items() if k != 'max_tokens'}) == {'input_tokens': len(rendered)}
    server.post('/slots/0?action=erase', {})
    reply = server.post('/v1/messages', body, headers={'x-api-key': 'unused', 'anthropic-version': '2023-06-01'})
    assert reply['type'] == 'message' and reply['role'] == 'assistant' and reply['model'] == 'claude-test' and reply['id'].startswith('msg_')
    assert reply['stop_reason'] == 'max_tokens' and reply['stop_sequence'] is None
    assert [block['type'] for block in reply['content']] == ['text']
    assert reply['usage'] == {'input_tokens': len(rendered), 'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0,
                              'cache_creation': {'ephemeral_5m_input_tokens': 0, 'ephemeral_1h_input_tokens': 0}, 'output_tokens': 9}
    text = reply['content'][0]['text']

    events = server.events('/v1/messages', {**body, 'stream': True})
    assert all(name == data['type'] for name, data in events)
    names = [name for name, _ in events]
    assert names[:3] == ['ping', 'message_start', 'content_block_start'] and names[-3:] == ['content_block_stop', 'message_delta', 'message_stop']
    assert set(names[3:-3]) == {'content_block_delta'}
    start = events[1][1]['message']
    # Without cache_control, the reused prefix is reported as read.
    assert start['content'] == [] and start['usage']['cache_read_input_tokens'] == len(rendered) and start['usage']['input_tokens'] == 0
    assert events[2][1] == {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}}
    deltas = [data['delta'] for name, data in events if name == 'content_block_delta']
    assert all(delta['type'] == 'text_delta' for delta in deltas) and ''.join(delta['text'] for delta in deltas) == text
    assert events[-2][1]['delta'] == {'stop_reason': 'max_tokens', 'stop_sequence': None} and events[-2][1]['usage']['output_tokens'] == 9

    # A stop sequence is reported by value; an assistant turn in the history is accepted.
    if len(text) >= 3:
        word = text[1:3]
        stopped = server.post('/v1/messages', {**body, 'stop_sequences': [word]})
        assert stopped['stop_reason'] == 'stop_sequence' and stopped['stop_sequence'] == word
        assert stopped['content'][0]['text'] == text[:text.find(word)]
    turns = body['messages'] + [{'role': 'assistant', 'content': [{'type': 'thinking', 'thinking': 't', 'signature': ''},
                                                                   {'type': 'text', 'text': 'Hello'}]},
                                {'role': 'user', 'content': 'More'}]
    assert server.post('/v1/messages', {**body, 'messages': turns})['usage']['output_tokens'] == 9

    for bad in [{'messages': body['messages']}, {'max_tokens': 5}, {'max_tokens': 5, 'messages': [{'role': 'system', 'content': 'x'}]},
                {'max_tokens': 5, 'messages': [{'role': 'user', 'content': [{'type': 'image', 'source': {}}]}]}]:
        error = server.post('/v1/messages', bad, 400)
        assert error['type'] == 'error' and error['error']['type'] == 'invalid_request_error' and error['error']['message']
    error = server.post('/v1/messages', {**body, 'messages': [{'role': 'user', 'content': 'x' * CONTEXT}]}, 400)
    assert error['type'] == 'error' and error['error']['type'] == 'invalid_request_error'


def chatml(messages):
    """ChatML as the server renders it for the dummy model (one byte per token)."""
    return ''.join(f'<|im_start|>{role}\n{content.strip()}<|im_end|>\n' for role, content in messages) + '<|im_start|>assistant\n'


def end_of(rendered, text, after=0):
    """Prompt position (byte offset) just after `text` in the rendered prompt."""
    return rendered.index(text, after) + len(text)


def openai_caching(server):
    """OpenAI prompt caching (GPT-5.6 semantics) with --cache-min-tokens 64."""
    system = ('You are a careful assistant. ' * 4).strip()
    first = 'Question one: ' + 'a' * 80
    body = {'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': first}], 'max_tokens': 4, 'temperature': 0}
    rendered = chatml([('system', system), ('user', first)])
    user_end = end_of(rendered, first)

    def details(reply):
        usage = reply['usage']
        assert usage['prompt_tokens'] + usage['completion_tokens'] == usage['total_tokens']
        return usage['prompt_tokens_details']['cached_tokens'], usage['prompt_tokens_details']['cache_write_tokens']

    # Implicit mode: a breakpoint at the end of the latest user message.
    reply = server.post('/v1/chat/completions', body)
    assert reply['usage']['prompt_tokens'] == len(rendered) and details(reply) == (0, user_end), reply['usage']
    assert details(server.post('/v1/chat/completions', body)) == (user_end, 0)
    # The next turn reads the previous user message's ending and writes only the new part.
    second = 'Question two: ' + 'b' * 50
    turn = {**body, 'messages': body['messages'] + [{'role': 'assistant', 'content': 'ok'}, {'role': 'user', 'content': second}]}
    rendered_turn = chatml([('system', system), ('user', first), ('assistant', 'ok'), ('user', second)])
    assert details(server.post('/v1/chat/completions', turn)) == (user_end, end_of(rendered_turn, second) - user_end)
    # Streaming reports the same accounting in the usage chunk.
    events = server.events('/v1/chat/completions', {**turn, 'stream': True, 'stream_options': {'include_usage': True}})
    usage = events[-2][1]['usage']['prompt_tokens_details']
    assert usage == {'cached_tokens': end_of(rendered_turn, second), 'cache_write_tokens': 0}, usage

    # Explicit mode: only the marked developer prefix is written and read; the
    # engine keeps a snapshot there, so a different question reuses it.
    reference = ('Reference material. ' * 6).strip()
    developer = {'role': 'developer', 'content': [{'type': 'text', 'text': reference, 'prompt_cache_breakpoint': {'mode': 'explicit'}}]}
    marked = {'prompt_cache_options': {'mode': 'explicit'}, 'messages': [developer, {'role': 'user', 'content': 'first question'}],
              'max_tokens': 3, 'temperature': 0}
    prefix = end_of(chatml([('system', reference)]), reference)
    assert details(server.post('/v1/chat/completions', marked)) == (0, prefix)
    other = {**marked, 'messages': [developer, {'role': 'user', 'content': 'second question'}]}
    reply = server.post('/v1/chat/completions', other)
    assert details(reply) == (prefix, 0) and reply['timings']['cache_n'] == prefix, (reply['usage'], reply['timings'])
    # prompt_cache_key separates cache accounting.
    keyed = {**other, 'prompt_cache_key': 'tenant-b'}
    assert details(server.post('/v1/chat/completions', keyed)) == (0, prefix)
    assert details(server.post('/v1/chat/completions', keyed)) == (prefix, 0)
    # A 12-hour breakpoint over 30-minute coverage is a new write, then reads.
    long_lived = {**other, 'messages': [{**developer, 'content': [{**developer['content'][0], 'prompt_cache_breakpoint': {'ttl': '12h'}}]},
                                        other['messages'][1]]}
    assert details(server.post('/v1/chat/completions', long_lived)) == (0, prefix)
    assert details(server.post('/v1/chat/completions', long_lived)) == (prefix, 0)
    # Explicit mode without breakpoints does not cache; prefixes below the minimum are not written.
    plain = {**body, 'prompt_cache_options': {'mode': 'explicit', 'ttl': '30m'}}
    assert details(server.post('/v1/chat/completions', plain)) == (0, 0)
    short = {'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 2}
    assert details(server.post('/v1/chat/completions', short)) == (0, 0) and details(server.post('/v1/chat/completions', short)) == (0, 0)

    for bad in [{'prompt_cache_options': {'ttl': '12h'}}, {'prompt_cache_options': {'mode': 'sometimes'}},
                {'prompt_cache_options': 'implicit'}, {'prompt_cache_retention': 'forever'},
                {'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'x', 'prompt_cache_breakpoint': {'ttl': '1h'}}]}]},
                {'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'x', 'prompt_cache_breakpoint': {'mode': 'implicit'}}]}]}]:
        error = server.post('/v1/chat/completions', {**body, **bad}, 400)['error']
        assert error['type'] == 'invalid_request_error' and 'prompt_cache' in error['message'], error
    assert details(server.post('/v1/chat/completions', {**short, 'prompt_cache_retention': '24h', 'prompt_cache_key': 'k'})) == (0, 0)


def anthropic_caching(server):
    """Anthropic prompt caching at cache_control breakpoints with --cache-min-tokens 64."""
    def usage(reply, read, created_5m=0, created_1h=0, total=None):
        value = reply['usage']
        expected = {'cache_read_input_tokens': read, 'cache_creation_input_tokens': created_5m + created_1h,
                    'cache_creation': {'ephemeral_5m_input_tokens': created_5m, 'ephemeral_1h_input_tokens': created_1h}}
        assert {k: value[k] for k in expected} == expected, value
        if total is not None:
            assert value['input_tokens'] == total - read - created_5m - created_1h, value

    system = ('Anthropic system prompt. ' * 5).strip()
    cached_system = [{'type': 'text', 'text': system, 'cache_control': {'type': 'ephemeral'}}]
    body = {'max_tokens': 3, 'temperature': 0, 'system': cached_system, 'messages': [{'role': 'user', 'content': 'Q1 please answer'}]}
    rendered = chatml([('system', system), ('user', 'Q1 please answer')])
    prefix = end_of(rendered, system)
    usage(server.post('/v1/messages', body), 0, prefix, total=len(rendered))
    reply = server.post('/v1/messages', {**body, 'messages': [{'role': 'user', 'content': 'Q2 something else'}]})
    usage(reply, prefix, total=len(chatml([('system', system), ('user', 'Q2 something else')])))
    # The streamed message_start carries the same usage.
    events = server.events('/v1/messages', {**body, 'stream': True, 'messages': [{'role': 'user', 'content': 'Q2 something else'}]})
    start = events[1][1]['message']
    usage(start, prefix)

    # Automatic caching: the top-level cache_control lands on the last block and
    # reads the system entry within the lookback window.
    question = 'Q3 ' + 'c' * 70
    automatic = {**body, 'cache_control': {'type': 'ephemeral'}, 'messages': [{'role': 'user', 'content': question}]}
    rendered = chatml([('system', system), ('user', question)])
    usage(server.post('/v1/messages', automatic), prefix, end_of(rendered, question) - prefix, total=len(rendered))
    usage(server.post('/v1/messages', automatic), end_of(rendered, question), total=len(rendered))

    # Mixed lifetimes: the 1-hour breakpoint comes first.
    other_system = ('A different system prompt that lives longer. ' * 2).strip()
    mixed = {'max_tokens': 2, 'system': [{'type': 'text', 'text': other_system, 'cache_control': {'type': 'ephemeral', 'ttl': '1h'}}],
             'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'Q4 ' + 'd' * 40, 'cache_control': {'type': 'ephemeral'}},
                                                       {'type': 'text', 'text': 'tail'}]}]}
    rendered = chatml([('system', other_system), ('user', 'Q4 ' + 'd' * 40 + '\ntail')])
    first_end, second_end = end_of(rendered, other_system), end_of(rendered, 'd' * 40)
    usage(server.post('/v1/messages', mixed), 0, second_end - first_end, first_end, total=len(rendered))

    # Pre-warming: max_tokens 0 evaluates the prompt only.
    warm_system = ('Warm this prefix before traffic arrives. ' * 3).strip()
    warm = {'max_tokens': 0, 'system': [{'type': 'text', 'text': warm_system, 'cache_control': {'type': 'ephemeral'}}],
            'messages': [{'role': 'user', 'content': 'warmup'}]}
    reply = server.post('/v1/messages', warm)
    warm_prefix = end_of(chatml([('system', warm_system)]), warm_system)
    assert reply['content'] == [] and reply['stop_reason'] == 'max_tokens' and reply['usage']['output_tokens'] == 0, reply
    usage(reply, 0, warm_prefix)
    usage(server.post('/v1/messages', {**warm, 'max_tokens': 2, 'messages': [{'role': 'user', 'content': 'real question'}]}), warm_prefix)
    # Prefixes below the minimum are not cached even with a breakpoint.
    tiny = {'max_tokens': 1, 'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'hi', 'cache_control': {'type': 'ephemeral'}}]}]}
    usage(server.post('/v1/messages', tiny), 0)
    usage(server.post('/v1/messages', tiny), 0)
    # Token counting accepts cache_control.
    assert server.post('/v1/messages/count_tokens', {k: v for k, v in mixed.items() if k != 'max_tokens'})['input_tokens'] == len(rendered)

    text = lambda value, **control: {'type': 'text', 'text': value, **({'cache_control': control} if control else {})}
    for bad, words in [
            ({'system': [text('a', type='ephemeral')], 'messages': [{'role': 'user', 'content': [text('b', type='ephemeral', ttl='1h')]}]},
             "ttl='1h'"),
            ({'messages': [{'role': 'user', 'content': [text(str(i), type='ephemeral') for i in range(5)]}]}, 'maximum of 4'),
            ({'messages': [{'role': 'user', 'content': [text('', type='ephemeral')]}]}, 'empty text'),
            ({'messages': [{'role': 'user', 'content': [text('x', type='persistent')]}]}, "'ephemeral'"),
            ({'messages': [{'role': 'user', 'content': [text('x', type='ephemeral', ttl='2h')]}]}, "'5m' or '1h'"),
            ({'messages': [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': [
                {'type': 'thinking', 'thinking': 't', 'signature': '', 'cache_control': {'type': 'ephemeral'}}]}]}, 'Extra inputs'),
            ({'cache_control': {'type': 'ephemeral', 'ttl': '1h'}, 'messages': [{'role': 'user', 'content': [text('x', type='ephemeral')]}]},
             'automatic'),
            ({'max_tokens': 0, 'stream': True, 'messages': [{'role': 'user', 'content': 'x'}]}, 'stream')]:
        error = server.post('/v1/messages', {'max_tokens': 2, **bad}, 400)
        assert error['type'] == 'error' and error['error']['type'] == 'invalid_request_error' and words in error['error']['message'], error

    metrics = server.get('/metrics')
    values = dict(line.split(' ') for line in metrics.splitlines() if line and not line.startswith('#'))
    assert int(values['tidal:cache_read_tokens_total']) > 0 and int(values['tidal:cache_write_tokens_total']) > 0
    assert int(values['tidal:cache_entries']) > 0 and int(values['tidal:cache_snapshots']) > 0

    # Erasing the slot drops the cache entries with the sequence.
    server.post('/slots/0?action=erase', {})
    usage(server.post('/v1/messages', body), 0, prefix)


def transport(server, reference):
    page = server.get('/')
    assert isinstance(page, str) and '<title>tidal-server</title>' in page and server.get('/index.html') == page

    # Keep-alive: several requests, a streamed one included, on one connection.
    connection = server.connection()
    for _ in range(3):
        connection.request('POST', '/tokenize', json.dumps({'content': 'ab'}), {'Content-Type': 'application/json'})
        response = connection.getresponse()
        assert response.status == 200 and json.loads(response.read()) == {'tokens': [97, 98]}
    connection.request('POST', '/completion', json.dumps({'prompt': 'ab', 'n_predict': 3, 'stream': True}))
    response = connection.getresponse()
    # One event per token and the final one; a trailing partial character adds one.
    assert response.status == 200 and response.getheader('Transfer-Encoding') == 'chunked' and response.read().count(b'data: ') in (4, 5)
    payload = json.dumps({'content': 'chunked'}).encode()
    connection.putrequest('POST', '/tokenize')
    connection.putheader('Transfer-Encoding', 'chunked')
    connection.endheaders()
    connection.send(b'3\r\n' + payload[:3] + b'\r\n%x\r\n' % (len(payload) - 3) + payload[3:] + b'\r\n0\r\n\r\n')
    response = connection.getresponse()
    assert response.status == 200 and json.loads(response.read()) == {'tokens': list(b'chunked')}
    connection.request('OPTIONS', '/v1/chat/completions')
    response = connection.getresponse()
    response.read()
    assert response.status == 204 and response.getheader('Access-Control-Allow-Origin') == '*'
    connection.close()

    # Concurrent requests queue for the single engine and are all answered.
    prompt = 'The quick brown fox'
    results = [None] * 6

    def worker(index):
        results[index] = server.post('/completion', {'prompt': prompt, 'n_predict': 12, 'temperature': 0, 'return_tokens': True})

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(results))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert all(result['tokens'] == reference['tokens'] for result in results)

    # A client that goes away mid-stream frees the engine.
    connection = server.connection()
    connection.request('POST', '/completion', json.dumps({'prompt': 'a', 'stream': True, 'temperature': 0}))
    connection.getresponse().read(64)
    connection.close()
    deadline = time.time() + 20
    while server.get('/slots')[0]['is_processing'] and time.time() < deadline:
        time.sleep(0.01)
    assert not server.get('/slots')[0]['is_processing']


def session_cache(binary, model, directory):
    """Conversations that take turns resume from the cache store (--cache-ram, --cache-disk)."""
    flags = ['--model', model, '--backend', 'cpu', '--cache-ram', '64M', '--cache-disk', '1G', '--cache-dir', directory,
             '--cache-chunk-tokens', '64', '--cache-branch-tokens', '64']
    system = [ord(c) for c in 'Shared instructions. ' + 'Be brief and precise. ' * 6]
    alpha = system + [ord(c) for c in 'First conversation: ' + 'alpha ' * 30]
    beta = system + [ord(c) for c in 'Second conversation: ' + 'beta ' * 25]
    gamma = system + [ord(c) for c in 'Third conversation: ' + 'gamma ' * 20]
    parting = len(system)  # The conversations part right after the shared instructions.
    more = [ord(c) for c in ' and then?']

    def ask(server, prompt, **extra):
        return server.post('/completion', {'prompt': prompt, 'n_predict': 6, 'temperature': 0, 'return_tokens': True, **extra})

    def store(server):
        return server.get('/slots')[0]['cache_store']

    def turns(server):
        """alpha, beta, alpha again after its reply, beta with another reply, a third conversation."""
        first = ask(server, alpha)
        second = ask(server, beta)
        third = ask(server, alpha + first['tokens'] + more)
        fourth = ask(server, beta + more)
        fifth = ask(server, gamma)
        return first, second, third, fourth, fifth

    server = Server(binary, *flags)
    plain = Server(binary, '--model', model, '--backend', 'cpu')
    try:
        assert server.port and plain.port, server.output() + plain.output()
        assert 'cache store of 64 MiB in memory and 1.0 GiB in ' + str(directory) in server.output(), server.output()
        assert plain.get('/slots')[0]['cache_store'] is None and plain.get('/props')['cache_store'] is None
        first = ask(server, alpha)
        assert first['tokens_cached'] == 0 and first['timings']['cache_store_n'] == 0
        # The sequence is saved in the background once the reply is out.
        deadline = time.time() + 30
        while store(server)['snapshots'] < 1 and time.time() < deadline:
            time.sleep(0.02)
        saved = store(server)
        assert saved['snapshots'] == 1 and saved['tokens'] == len(alpha) // 64 * 64 and saved['ram_bytes'] > 0, saved
        assert saved['ram_limit'] == 64 << 20 and saved['disk_limit'] == 1 << 30 and saved['chunk_tokens'] == 64
        # Another conversation takes the engine. It shares the instructions with
        # the first, so it leaves a snapshot where the two part.
        second = ask(server, beta)
        assert second['tokens_cached'] == 0
        # The first conversation returns after its reply: nothing up to the end
        # of that reply is evaluated. The engine still holds the instructions
        # (cache_n counts all reuse); the rest comes from the store (cache_store_n).
        # The last token of a reply is sampled but not consumed, so it is evaluated with the new turn.
        turn = alpha + first['tokens'] + more
        third = ask(server, turn)
        resumed = len(alpha) + len(first['tokens']) - 1
        assert third['tokens_cached'] == resumed and third['timings']['cache_store_n'] == resumed - parting, third['timings']
        assert third['timings']['prompt_n'] == len(more) + 1
        # The second returns with a different continuation: its prompt end is restored.
        fourth = ask(server, beta + more)
        assert fourth['tokens_cached'] == len(beta) and fourth['timings']['cache_store_n'] == len(beta) - parting, fourth['timings']
        # A third conversation resumes where the others parted.
        fifth = ask(server, gamma)
        assert fifth['tokens_cached'] == parting, (fifth['tokens_cached'], parting)
        # Stored state gives the replies of a server that evaluates everything.
        expected = turns(plain)
        for ours, theirs in zip([first, second, third, fourth, fifth], expected):
            assert ours['tokens'] == theirs['tokens'], (ours['tokens'], theirs['tokens'])
        assert [reply['tokens_cached'] for reply in expected] == [0, 0, 0, 0, 0]
        # cache_prompt false evaluates everything and keeps the store.
        assert ask(server, turn, cache_prompt=False)['tokens_cached'] == 0
        again = ask(server, turn)
        assert again['tokens_cached'] == len(turn) and again['tokens'] == third['tokens']
        values = dict(line.split(' ') for line in server.request('GET', '/metrics')[1].splitlines() if not line.startswith('#'))
        assert int(values['tidal:cache_store_restores_total']) == 2 and int(values['tidal:cache_store_snapshots']) >= 3, values
        assert int(values['tidal:cache_store_restored_tokens_total']) >= resumed + len(beta) - 2 * parting
        assert server.get('/props')['cache_store']['directory'] == str(directory)
    except BaseException:
        print(server.output()[-3000:])
        server.stop(); plain.stop()
        raise
    assert plain.stop() == 0
    # An orderly shutdown writes what memory holds; the next run resumes from the files.
    assert server.stop() == 0
    server = Server(binary, *flags)
    try:
        assert server.port, server.output()
        adopted = store(server)
        assert adopted['snapshots'] >= 3 and adopted['disk_bytes'] > 0 and adopted['ram_bytes'] == 0, adopted
        reply = ask(server, beta + more)
        assert reply['tokens_cached'] == len(beta) + len(more) and reply['tokens'] == fourth['tokens'], reply['timings']
        assert store(server)['read_bytes'] > 0
        # Erasing the slot empties the store.
        server.post('/slots/0?action=erase', {})
        emptied = store(server)
        assert emptied['snapshots'] == 0 and emptied['ram_bytes'] == 0 and emptied['disk_bytes'] == 0, emptied
        assert not list(Path(directory).rglob('*.tkv'))
        assert ask(server, beta + more)['tokens_cached'] == 0
    except BaseException:
        print(server.output()[-3000:])
        server.stop()
        raise
    assert server.stop() == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bin-dir', type=Path, required=True)
    args = parser.parse_args()
    binary = args.bin_dir / 'tidal-server'
    with tempfile.TemporaryDirectory(prefix='tidal-server-') as directory:
        model = Path(directory) / 'unit-model'
        subprocess.run([str(args.bin_dir / 'tidal-dummy'), '--output', str(model), '--context', str(CONTEXT), '--type', 'f32'],
                       check=True, capture_output=True, timeout=120)
        server = Server(binary, '--model', model, '--backend', 'cpu')
        try:
            assert server.port, server.output()
            reference = llama_endpoints(server)
            openai_endpoints(server, reference)
            anthropic_endpoints(server)
            transport(server, reference)
        except BaseException:
            print(server.output()[-3000:])
            server.stop()
            raise
        assert server.stop() == 0, 'SIGTERM must end the server cleanly'
        print('endpoints, streaming, prompt cache, queuing and shutdown passed', flush=True)

        # API prompt caching needs longer prompts than the main model's context.
        cache_model = Path(directory) / 'cache-model'
        subprocess.run([str(args.bin_dir / 'tidal-dummy'), '--output', str(cache_model), '--context', '1024', '--type', 'f32'],
                       check=True, capture_output=True, timeout=120)
        server = Server(binary, '--model', cache_model, '--backend', 'cpu', '--cache-min-tokens', '64', '--cache-snapshots', '8')
        try:
            assert server.port, server.output()
            openai_caching(server)
            anthropic_caching(server)
        except BaseException:
            print(server.output()[-3000:])
            server.stop()
            raise
        assert server.stop() == 0
        print('OpenAI and Anthropic prompt caching passed', flush=True)

        session_cache(binary, cache_model, Path(directory) / 'cache-store')
        print('cache store across conversations and restarts passed', flush=True)

        # Defaults and authentication come from the command line.
        server = Server(binary, '--model', model, '--backend', 'cpu', '--api-key', 'secret', '--alias', 'named',
                        '--temp', '0', '--n-predict', '5', '--webui', '0', '--context', '64')
        try:
            assert server.port, server.output()
            assert server.request('GET', '/health')[0] == 200 and server.request('GET', '/')[0] == 404
            assert server.request('GET', '/v1/models')[1]['data'][0]['id'] == 'named'
            code, data = server.request('POST', '/completion', {'prompt': 'x'})
            assert code == 401 and data['error']['type'] == 'authentication_error'
            code, data = server.request('POST', '/v1/messages', {'max_tokens': 1, 'messages': []}, {'x-api-key': 'wrong'})
            assert code == 401 and data == {'type': 'error', 'error': {'type': 'authentication_error', 'message': 'Invalid API key'}}
            first = server.post('/completion', {'prompt': 'x', 'return_tokens': True}, headers={'Authorization': 'Bearer secret'})
            second = server.post('/completion', {'prompt': 'x', 'return_tokens': True}, headers={'x-api-key': 'secret'})
            assert first['tokens_predicted'] == 5 and first['tokens'] == second['tokens']
            assert first['generation_settings']['temperature'] == 0 and first['generation_settings']['n_predict'] == 5
            props = server.request('GET', '/props', headers={'x-api-key': 'secret'})[1]
            assert props['default_generation_settings']['n_ctx'] == 64 and props['model_alias'] == 'named'
        except BaseException:
            print(server.output()[-3000:])
            server.stop()
            raise
        assert server.stop() == 0
        print('command-line defaults and API key passed', flush=True)

        # Startup failures exit with an error instead of serving.
        for flags in [['--model', Path(directory) / 'missing.gguf', '--backend', 'cpu'], ['--backend', 'cpu'],
                      ['--model', model, '--backend', 'cpu', '--thinking', 'maybe'],
                      ['--model', model, '--backend', 'cpu', '--temp', '-2'],
                      ['--model', model, '--backend', 'cpu', '--cache-min-tokens', '0'],
                      ['--model', model, '--backend', 'cpu', '--cache-disk', '1G'],
                      ['--model', model, '--backend', 'cpu', '--cache-dir', directory],
                      ['--model', model, '--backend', 'cpu', '--cache-ram', 'lots'],
                      ['--model', model, '--backend', 'cpu', '--cache-branch-tokens', '64']]:
            failed = Server(binary, *flags)
            assert failed.process.wait(timeout=60) == 1, flags
            assert 'tidal-server: ' in failed.output()
            failed.stop()
        print('startup failures passed', flush=True)


if __name__ == '__main__':
    main()
