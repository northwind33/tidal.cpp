#!/usr/bin/env python3
"""Persistent CUDA KV: independent equations, state transitions and placement."""
import argparse
import copy
import json
import sys
import subprocess
import tempfile
from pathlib import Path
from test_reference import run, PROMPT, Reference, np
from test_qwen35 import fixture, write_gguf
from qwen35_reference import Qwen35Reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bin-dir', type=Path, required=True)
    b = parser.parse_args().bin_dir
    with tempfile.TemporaryDirectory(prefix='tidal-gpu-kv-') as directory:
        root = Path(directory)
        fixtures = []
        for kind in ['f32', 'f16', 'bf16', 'q8_0', 'q4_0']:
            model = root / f'qsa-{kind}.gguf'
            run(b/'tidal-dummy', '--output', model, '--context', 64, '--type', kind)
            fixtures.append((model, Reference, .025 if kind.startswith('q') else .004, kind == 'f32'))
        for name, flags in [('ple', ['--ple-layers', '0,2']),
                            ('moe-ple', ['--experts', 4, '--experts-used', 2, '--ple-layers', '0,2'])]:
            model = root / f'{name}.gguf'
            run(b/'tidal-dummy', '--output', model, '--context', 64, '--type', 'f32', *flags)
            fixtures.append((model, Reference, .004, True))
        for tied in [True, False]:
            model = root / f'qwen35-{tied}.gguf'
            write_gguf(model, *fixture(tied=tied, explicit=True, mtp=True))
            fixtures.append((model, Qwen35Reference, .004, True))
        wide = root/'wide-qsa.gguf'
        run(b/'tidal-dummy', '--preset', 'dense-27b', '--hidden', 256, '--ffn', 512,
            '--layers', 4, '--vocab', 256, '--context', 4352, '--output', wide, '--type', 'f32')
        fixtures.append((wide, Reference, .004, True))
        for model, oracle, tolerance, state_test in fixtures:
            for precision in ['fp16', 'bf16']:
                if state_test:
                    run(b/'tidal-gpu-kv-test', model, precision)
                report = root / f'{model.stem}-{precision}.json'
                run(b/'tidal-cli', '--model', model, '--backend', 'cuda', '--kv-placement', 'gpu',
                    '--prefill-precision', precision, '--tokens', ','.join(map(str, PROMPT)),
                    '--steps', 3, '--repeat', 2, '--json', report)
                data = json.loads(report.read_text())
                assert data['kv_placement'] == 'gpu' and data['attention_backend'] == 'CUDA'
                assert data['kv_type'] == ('F16' if precision == 'fp16' else 'BF16')
                stats = data['statistics']
                assert stats['gpu_kv_bytes'] > 0 and stats['gpu_attention_calls'] > 0
                assert not stats['full_attention_calls'] and not stats['sparse_attention_calls']
                assert stats['prefill_kv_to_host_bytes'] == 0
                reference = oracle(model, 'cuda', kv_type=precision)
                reference.gpu_attention = True
                for i, token in enumerate(PROMPT+data['generated']):
                    reference.prefill_columns = len(PROMPT) if i < len(PROMPT) else 0
                    reference.prefill_head_columns = reference.prefill_columns if i != len(PROMPT)-1 else 0
                    expected, selected = reference.decode(token)
                    actual = np.asarray(data['trace'][i]['logits'])
                    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance,
                                               err_msg=f'{model.stem}/{precision} position {i}')
                    assert np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),1e-12) < tolerance
                    assert data['trace'][i]['selections'] == selected
                    if len(PROMPT)-1 <= i < len(PROMPT)+2:
                        assert int(expected.argmax()) == data['generated'][i-len(PROMPT)+1]
            print(f'{model.stem}: FP16/BF16 GPU KV oracle, QSA/full attention and continuation passed', flush=True)
        for name in ['long-qsa-ple', 'long-qwen35']:
            model = root / f'{name}.gguf'
            if name == 'long-qsa-ple':
                run(b/'tidal-dummy', '--output', model, '--context', 768, '--type', 'f32', '--ple-layers', '0,2')
            else:
                metadata, tensors = fixture(tied=False, explicit=True)
                metadata['qwen35.context_length'] = (4,768)
                write_gguf(model, metadata, tensors)
            for precision in ['fp16', 'bf16']:
                run(b/'tidal-gpu-kv-test', model, precision)
            print(f'{name}: GPU KV padding, decode-to-prefill, prefix extension and reset passed', flush=True)
        model = fixtures[0][0]
        for args in [('--kv-placement', 'invalid'), ('--backend', 'cpu', '--kv-placement', 'gpu')]:
            assert run(b/'tidal-cli', '--model', model, *args, ok=False).returncode == 1
        bench = root/'gpu.bench.json'
        run(b/'tidal-bench', '--model', model, '--kv-placement', 'gpu', '--prompt', 24,
            '--steps', 3, '--warmup', 0, '--repetitions', 2, '--profile', 'phases', '--json', bench)
        result = json.loads(bench.read_text())
        assert result['workload']['prefill']['kv_placement'] == 'gpu'
        assert result['workload']['prefill']['decode_attention_backend'] == 'CUDA'
        for item in result['runs']:
            assert item['statistics']['gpu_attention_calls'] > 0
            assert item['statistics']['prefill_kv_to_host_bytes'] == 0
            assert item['statistics']['d2h_bytes'] == 4+3*(4+12)
        changed = copy.deepcopy(result)
        changed['workload']['prefill']['kv_placement'] = 'cpu'
        incompatible = root/'changed.bench.json'
        incompatible.write_text(json.dumps(changed))
        comparison = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1]/'tools/compare_bench.py'),
                                     str(bench), str(incompatible)], capture_output=True, text=True, timeout=120)
        assert comparison.returncode == 2 and 'KV placement' in comparison.stderr
        print('GPU KV reporting, compact transfers, profiling replay and incompatible comparison rejection passed')


if __name__ == '__main__':
    main()
