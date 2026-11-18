#!/usr/bin/env python3
"""Cache store round trips: sequences that leave the engine come back bit-identical."""
import argparse
import tempfile
from pathlib import Path
from test_reference import run
from test_qwen35 import fixture, write_gguf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bin-dir', type=Path, required=True)
    parser.add_argument('--backend', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    b = args.bin_dir
    with tempfile.TemporaryDirectory(prefix='tidal-session-cache-') as directory:
        root = Path(directory)
        # QSA index rows, PLE convolution history and MoE routing all travel with a sequence.
        for name, flags in [('dense', []), ('ple', ['--ple-layers', '0,2']),
                            ('moe-ple', ['--experts', 4, '--experts-used', 2, '--ple-layers', '0,2'])]:
            model = root / (name + '.gguf')
            run(b / 'tidal-dummy', '--output', model, '--context', 96, '--type', 'f32', *flags)
            run(b / 'tidal-session-cache-test', model, args.backend)
            print(name + ': session cache passed', flush=True)
        for mtp in [False, True]:
            metadata, tensors = fixture(tied=False, explicit=True, mtp=mtp)
            metadata['qwen35.context_length'] = (4, 96)
            model = root / f'qwen35-{mtp}.gguf'
            write_gguf(model, metadata, tensors)
            run(b / 'tidal-session-cache-test', model, args.backend)
            if mtp and args.backend == 'cuda':
                run(b / 'tidal-session-cache-test', model, args.backend, 'mtp')
            print(f'qwen35 mtp={mtp}: session cache passed', flush=True)


if __name__ == '__main__':
    main()
