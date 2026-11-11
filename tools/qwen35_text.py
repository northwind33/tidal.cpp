#!/usr/bin/env python3
"""Optional text frontend: tokenization only; tidal-cli performs all inference."""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True, help='Official Qwen3.5-0.8B tokenizer.json')
    parser.add_argument('--bin-dir', type=Path, default=Path('build/dev-cuda/bin'))
    parser.add_argument('--backend', choices=['cuda','cpu'], default='cuda')
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--context', type=int, default=4096)
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--raw', action='store_true', help='Use the prompt literally, without single-turn ChatML framing')
    parser.add_argument('--thinking', action='store_true', help='Allow the model to generate its thinking section')
    parser.add_argument('--sampler', choices=['device', 'cpu'], default='device')
    for name, default in [('temp', 0.), ('top-p', 1.), ('min-p', 0.), ('repeat-penalty', 1.),
                          ('frequency-penalty', 0.), ('presence-penalty', 0.)]:
        parser.add_argument('--'+name, type=float, default=default)
    parser.add_argument('--top-k', type=int, default=0)
    parser.add_argument('--penalty-last-n', type=int, default=64)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--prefill-precision',choices=['fp16','bf16'],default='fp16')
    parser.add_argument('--kv-placement',choices=['cpu','gpu'],default='cpu')
    parser.add_argument('--prefill-chunk',default='auto')
    parser.add_argument('--prefill-tile',default='auto')
    parser.add_argument('--prefill-memory-mib',type=int,default=0)
    parser.add_argument('--prefill-reserve-mib',default='auto')
    args = parser.parse_args()
    sampling = [part for name in ['sampler','temp','top_k','top_p','min_p','repeat_penalty',
                                  'frequency_penalty','presence_penalty','penalty_last_n','seed',
                                  'kv_placement','prefill_precision','prefill_chunk','prefill_tile','prefill_memory_mib','prefill_reserve_mib']
                for part in ['--'+name.replace('_','-'), str(getattr(args,name))]]
    try:
        from tokenizers import Tokenizer
    except ImportError:
        parser.error('Install the optional text dependency: python3 -m pip install tokenizers')
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    prompt = args.prompt
    if not args.raw:
        prompt = '<|im_start|>user\n' + prompt + '<|im_end|>\n<|im_start|>assistant\n'
        if not args.thinking: prompt += '<think>\n\n</think>\n\n'
    tokens = tokenizer.encode(prompt, add_special_tokens=False).ids
    if not tokens: parser.error('Prompt must tokenize to at least one token')
    with tempfile.TemporaryDirectory(prefix='tidal-text-') as directory:
        trace = Path(directory)/'trace.json'
        subprocess.run([str(args.bin_dir/'tidal-cli'), '--model',str(args.model),'--backend',args.backend,
                        '--context',str(args.context),'--tokens',','.join(map(str,tokens)),
                        '--steps',str(args.steps),'--stop-at-eos','1','--json',str(trace),'--logits','0',*sampling],
                       check=True, stdout=subprocess.DEVNULL)
        result = json.loads(trace.read_text())
        if result['architecture'] != 'qwen35': parser.error('This frontend requires a native qwen35 GGUF')
        print(tokenizer.decode(result['generated'], skip_special_tokens=True))


if __name__ == '__main__': main()
