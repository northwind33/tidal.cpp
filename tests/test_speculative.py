#!/usr/bin/env python3
"""Speculative verification (external drafts) and MTP drafting state checks."""
import argparse
import tempfile
from pathlib import Path
from test_reference import run
from test_qwen35 import fixture, write_gguf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bin-dir', type=Path, required=True)
    b = parser.parse_args().bin_dir
    with tempfile.TemporaryDirectory(prefix='tidal-speculative-') as directory:
        root = Path(directory)
        qsa = root / 'qsa.gguf'
        run(b / 'tidal-dummy', '--output', qsa, '--context', 64, '--type', 'f32')
        run(b / 'tidal-speculative-test', qsa)
        quantized = root / 'qsa-q8.gguf'
        run(b / 'tidal-dummy', '--output', quantized, '--context', 96, '--type', 'q8_0')
        run(b / 'tidal-speculative-test', quantized)
        for name, context in [('qwen35', 64), ('qwen35-long', 768)]:
            metadata, tensors = fixture(tied=name == 'qwen35-long', explicit=True, mtp=True)
            metadata['qwen35.context_length'] = (4, context)
            model = root / f'{name}.gguf'
            write_gguf(model, metadata, tensors)
            run(b / 'tidal-speculative-test', model)
            run(b / 'tidal-speculative-test', model, 'mtp')
        print('speculative verification and MTP drafting passed', flush=True)


if __name__ == '__main__': main()
