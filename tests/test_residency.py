#!/usr/bin/env python3
"""Weight eviction/reload coverage across architectures and storage policies."""
import argparse
import tempfile
from pathlib import Path
from test_reference import run
from test_qwen35 import fixture, write_gguf


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bin-dir', required=True, type=Path)
    p.add_argument('--backend', choices=['cpu','cuda'], default='cpu')
    a = p.parse_args()
    with tempfile.TemporaryDirectory(prefix='tidal-residency-') as directory:
        root = Path(directory)
        for label, flags in [('f32', []), ('bf16', ['--type','bf16']), ('q4', ['--type','q4_0']),
                             ('moe-ple', ['--type','q8_0','--experts',4,'--experts-used',2,'--ple-layers','0,2'])]:
            model = root / (label+'.gguf')
            run(a.bin_dir/'tidal-dummy','--output',model,'--context',32,*flags)
            run(a.bin_dir/'tidal-residency-test', model, a.backend)
        for tied in [True, False]:
            model = root / f'qwen35-{tied}.gguf'
            metadata, tensors = fixture(tied=tied, explicit=True, mtp=True)
            write_gguf(model, metadata, tensors)
            run(a.bin_dir/'tidal-residency-test', model, a.backend)
    print(f'{a.backend}: Dense/MoE/PLE/qwen35 weight residency and exact asynchronous reload passed')


if __name__ == '__main__': main()
