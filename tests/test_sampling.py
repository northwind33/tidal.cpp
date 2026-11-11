#!/usr/bin/env python3
"""Independent fixed-logit distribution oracle and end-to-end sampling checks."""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

DEFAULT = dict(temp=0., top_k=0, top_p=1., min_p=0., repeat_penalty=1.,
               frequency_penalty=0., presence_penalty=0., penalty_last_n=64, seed=0)


def run(*args, ok=True):
    p = subprocess.run(list(map(str, args)), text=True, capture_output=True, timeout=120)
    assert (p.returncode == 0) == ok, (args, p.returncode, p.stdout[-1000:], p.stderr[-2000:])
    return p


def uniforms(seed):
    mask = (1 << 64) - 1
    while True:
        seed = (seed + 0x9e3779b97f4a7c15) & mask
        z = ((seed ^ (seed >> 30)) * 0xbf58476d1ce4e5b9) & mask
        z = ((z ^ (z >> 27)) * 0x94d049bb133111eb) & mask
        yield ((z ^ (z >> 31)) >> 11) * 2.**-53


def oracle(logits, history, cfg, u):
    c = DEFAULT | cfg
    c = {k: np.float32(v) if isinstance(v, float) else v for k, v in c.items()}
    v = np.asarray(logits, dtype=np.float32).copy()
    window = c['penalty_last_n']
    recent = history if window == -1 else history[-window:] if window else []
    count = np.bincount(recent, minlength=len(v))
    for token in np.flatnonzero(count):
        v[token] = v[token] / c['repeat_penalty'] if v[token] > 0 else v[token] * c['repeat_penalty']
        v[token] -= c['presence_penalty']
        v[token] -= c['frequency_penalty'] * np.float32(count[token])
    if c['temp'] == 0:
        token = int(np.argmax(v))
        return token, np.array([token]), np.array([1.]), 1
    ids = np.argsort(-v, kind='stable')
    if c['top_k'] > 0: ids = ids[:c['top_k']]
    mass = np.exp((v[ids].astype(np.float64) - float(v[ids[0]])) / float(c['temp']))
    mass[mass < c['min_p']] = 0
    cdf = np.cumsum(mass)
    kept = int(np.searchsorted(cdf, float(c['top_p']) * cdf[-1], side='left')) + 1
    threshold = min(u * cdf[kept - 1], np.nextafter(cdf[kept - 1], 0))
    return int(ids[np.searchsorted(cdf[:kept], threshold, side='right')]), ids, cdf, kept


def flags(c):
    return [item for k, v in c.items() for item in ('--' + k.replace('_', '-'), str(v))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bin-dir', type=Path, required=True)
    p.add_argument('--backend', choices=['cpu', 'cuda'], default='cpu')
    args = p.parse_args()
    probe, cli, dummy, bench = [args.bin_dir / name for name in ('tidal-sampling-test','tidal-cli','tidal-dummy','tidal-bench')]
    rng = np.random.default_rng(71)
    # Independent golden RNG values, including unsigned overflow.
    g = uniforms(0)
    np.testing.assert_array_equal([next(g), next(g)], [0.8833108082136426, 0.43152799704850997])
    assert next(uniforms(7046029254386353131)) == 0
    assert next(uniforms(3558559446808474027)) == 1 - 2.**-53
    with tempfile.TemporaryDirectory(prefix='tidal-sampling-') as directory:
        root = Path(directory)
        cases = [
            ([-2, 1, 1, -3], {}, []),
            ([-np.finfo(np.float32).max] * 5, {}, []),
            ([-0., 0., -0., 0.], dict(temp=1.), []),
            ([3, 2, 1, 0, -1], dict(temp=.7, top_k=3, top_p=.8), []),
            ([3, 2, 1], dict(repeat_penalty=2.), [0,0]),
            ([1, 0, -1000], dict(temp=1., seed=7046029254386353131), []),
            ([1, 0, -1000], dict(temp=1., seed=3558559446808474027), []),
            ([0, 0, 0, 0], dict(temp=1., top_p=.5), []),
            ([3, 2, 1, 0, -1], dict(temp=2., min_p=.5, top_p=.8), []),
            ([3, 3, 1, -2], dict(temp=1., min_p=1., top_k=-1), []),
            ([3, 3, 1, -2], dict(temp=1., top_k=1), []),
            ([3, 3, 1, -2], dict(temp=1., top_k=100, top_p=1e-10), []),
            ([-3e38, 3e38, 3e38], dict(temp=1e-20), []),
            ([2, -2, 1, -1, .5], dict(temp=.9, repeat_penalty=1.3, frequency_penalty=.4,
                                     presence_penalty=.2, penalty_last_n=3, seed=18446744073709551615), [0,1,0,3,0]),
            ([2, -2, 1, -1, .5], dict(temp=.9, repeat_penalty=.8, frequency_penalty=-.4,
                                     presence_penalty=-.2, penalty_last_n=-1, seed=42), [0,1,0,3,0]),
            ([2, -2, 1, -1, .5], dict(repeat_penalty=2., penalty_last_n=0), [0,0]),
            (rng.normal(size=1031), dict(temp=.8, top_k=47, top_p=.95, min_p=.01, seed=123), []),
        ]
        large = rng.normal(size=248320).astype(np.float32)
        large[[1, 1024, 248319]] = 100
        cases += [(large, {}, []), (large, dict(temp=10., top_k=64, top_p=.9), []),
                  (large, dict(temp=10., top_p=.9), [])]
        for index, (values, config, history) in enumerate(cases):
            values = np.asarray(values, dtype=np.float32)
            path = root / 'logits.bin'; values.tofile(path)
            n = 12 if len(values) < 2000 else 1
            for sampler in ['device', 'cpu']:
                result = json.loads(run(probe, '--input', path, '--backend', args.backend, '--sampler', sampler,
                    '--history', ','.join(map(str,history)), '--draws', n, '--feedback', 1, *flags(config)).stdout)
                assert result[0] == result[1], ('reset', index, sampler)
                consumed = list(history); draws = uniforms(config.get('seed', 0))
                for actual in result[0]:
                    token, ids, cdf, kept = oracle(values, consumed, config, next(draws))
                    assert actual['token'] == token and actual['kept'] == kept, (index, sampler, actual['token'], token)
                    np.testing.assert_array_equal(actual['ids'], ids)
                    np.testing.assert_allclose(actual['cdf'], cdf, rtol=3e-13, atol=1e-15)
                    consumed.append(token)
        # A reproducible empirical distribution test, without feedback penalties.
        path = root / 'logits.bin'; np.array([0., 1., 2.], np.float32).tofile(path)
        out = json.loads(run(probe, '--input', path, '--backend', args.backend, '--temp', 1,
                             '--draws', 4096, '--diagnostics', 0).stdout)
        hist = np.bincount([x['token'] for x in out[0]], minlength=3) / 4096
        np.testing.assert_allclose(hist, np.exp([0.,1.,2.]) / np.exp([0.,1.,2.]).sum(), atol=.025, rtol=0)
        for bad in [dict(temp=-1), dict(temp='nan'), dict(top_k=-2), dict(top_p=0), dict(top_p=1.1),
                    dict(min_p=-.1), dict(repeat_penalty=0), dict(penalty_last_n=-2), dict(seed=-1), dict(seed=1<<64)]:
            run(probe, '--input', path, '--backend', args.backend, *flags(bad), ok=False)
        for bad in [np.nan, np.inf, -np.inf]:
            np.array([0., bad], np.float32).tofile(path)
            for sampler in ['cpu','device']:
                run(probe, '--input', path, '--backend', args.backend, '--sampler', sampler, ok=False)
        np.array([np.finfo(np.float32).max, 0], np.float32).tofile(path)
        for sampler in ['cpu','device']:
            run(probe, '--input', path, '--backend', args.backend, '--sampler', sampler,
                '--history', '0', '--repeat-penalty', .01, ok=False)
        # Same neural backend, independently recomputed sampler decisions. JSON
        # capture and generation-only mode must preserve RNG and penalty history.
        model = root / 'model.gguf'; run(dummy, '--output', model, '--context', 32)
        prompt = [1,2,7,2]
        config = dict(temp=.8, top_k=31, top_p=.92, min_p=.01, repeat_penalty=1.2,
                      frequency_penalty=.1, presence_penalty=.3, penalty_last_n=6, seed=71)
        outputs = []
        for sampler in ['device','cpu']:
            for capture in [1,0]:
                report = root / 'trace.json'
                run(cli, '--model', model, '--backend', args.backend, '--sampler', sampler,
                    '--tokens', ','.join(map(str,prompt)), '--steps', 9, '--repeat', 2,
                    '--logits', capture, '--json', report, *flags(config))
                data = json.loads(report.read_text()); outputs.append(data['generated'])
                assert data['statistics']['sampling_calls'] == 9
                if capture:
                    history = prompt.copy(); draws = uniforms(config['seed'])
                    for i, actual in enumerate(data['generated']):
                        logits = data['trace'][len(prompt)-1+i]['logits']
                        expected, *_ = oracle(logits, history, config, next(draws))
                        assert actual == expected
                        history.append(actual)
                else:
                    assert all(not s['logits'] for s in data['trace'])
                if args.backend == 'cuda':
                    s = data['statistics']
                    expected = 9*12 if sampler == 'device' else 0 if capture else 9*256*4
                    assert s['sampling_device_to_host_bytes'] == expected, s
        assert all(x == outputs[0] for x in outputs)
        # Corrupt only the output matrix: compact prompt ingestion must still
        # reject nonfinite head outputs even when no token is sampled.
        from reference import read_gguf
        _, weights, _, _ = read_gguf(model)
        offset = model.read_bytes().find(weights['output.weight'].astype('<f4').tobytes())
        assert offset >= 0
        corrupt = root / 'nonfinite.gguf'
        corrupt.write_bytes(model.read_bytes())
        with corrupt.open('r+b') as f:
            f.seek(offset)
            f.write(np.float32(np.nan).tobytes())
        for capture in [0,1]:
            result = run(cli, '--model', corrupt, '--backend', args.backend, '--tokens', '1',
                         '--steps', 0, '--logits', capture, ok=False)
            assert 'Non-finite logits' in result.stderr, result.stderr
        # Normal benchmark timings include sampling without full-logit readback.
        report = root / 'bench.json'
        run(bench, '--model', model, '--backend', args.backend, '--tokens', '1,2', '--steps', 3,
            '--warmup', 0, '--repetitions', 1, '--profile', 'phases', '--json', report, *flags(config))
        b = json.loads(report.read_text())
        assert not b['workload']['capture_logits'] and b['profile']['logits_max_abs_vs_baseline'] is None
        assert b['profile']['sampled_matches_baseline']
        events = [e for e in b['profile']['events'] if e['args']['phase'] == 'sampling']
        assert len([e for e in events if e['cat'] == 'sampler']) == 3
        if args.backend == 'cuda':
            assert {'sort','cumulative_sum','top_p_draw'} <= {e['name'] for e in events}
            assert all(e['args']['backend'].startswith('CUDA') for e in events)
            for direction, field in [('D2H','sampling_device_to_host_bytes'), ('H2D','sampling_host_to_device_bytes')]:
                assert sum(e['args']['bytes'] for e in events if e['name'] == direction) == b['profile']['statistics'][field]
    print(f'{args.backend}: sampling distributions, boundaries, penalties, RNG/reset, errors, transfers and generation passed')


if __name__ == '__main__': main()
