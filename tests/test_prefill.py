#!/usr/bin/env python3
"""Independent batch arithmetic and layer-major prefill state/placement checks."""
import argparse
import json
import tempfile
from pathlib import Path
from test_reference import run, Reference, PROMPT, np
from test_qwen35 import fixture, write_gguf
from qwen35_reference import Qwen35Reference


def batch_matrices(binary, model, backend):
    ref = Reference(model, backend)
    ref.prefill_columns = 24 if backend == 'cuda' else 0
    rng = np.random.default_rng(41)
    requests, expected = [], []
    for name in ['blk.0.hc_attn.down', 'blk.0.gdn.qkv', 'blk.0.ffn.up', 'output.weight']:
        width = ref.w[name].shape[-1]
        values = rng.uniform(-2, 2, (24,width)).astype(np.float32)
        values[0] = 0
        requests.append('@batch '+name+' 24 '+' '.join(map(str,values.reshape(-1))))
        expected.append(np.stack([ref.mm(name, row) for row in values]))
    output = run(binary, model, backend, input='\n'.join(requests)+'\n').stdout.splitlines()
    assert len(output) == len(expected)
    for line, target in zip(output, expected):
        fields = line.split()
        actual = np.asarray(fields[1:],dtype=np.float32).reshape(target.shape)
        np.testing.assert_allclose(actual,target,atol=3e-5,rtol=3e-5,err_msg=f'batch GEMM {fields[0]}')


def trajectory(binary, model, output, backend, reference, tolerance):
    run(binary,'--model',model,'--backend',backend,'--tokens',','.join(map(str,PROMPT)),
        '--steps',3,'--repeat',2,'--json',output)
    data=json.loads(output.read_text())
    for i,token in enumerate(PROMPT+data['generated']):
        if hasattr(reference,'prefill_columns'):
            reference.prefill_columns = len(PROMPT) if backend=='cuda' and i<len(PROMPT) else 0
            reference.prefill_head_columns = reference.prefill_columns if i!=len(PROMPT)-1 else 0
        expected,selected=reference.decode(token)
        actual=np.asarray(data['trace'][i]['logits'])
        np.testing.assert_allclose(actual,expected,atol=tolerance,rtol=tolerance,err_msg=f'prefill/continuation position {i}')
        assert np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),1e-12)<tolerance
        assert data['trace'][i]['selections']==selected
        if len(PROMPT)-1<=i<len(PROMPT)+2:
            assert int(expected.argmax())==data['generated'][i-len(PROMPT)+1]
    return data


def main():
    p=argparse.ArgumentParser();p.add_argument('--bin-dir',type=Path,required=True)
    p.add_argument('--backend',choices=['cpu','cuda'],default='cpu');a=p.parse_args();b=a.bin_dir
    with tempfile.TemporaryDirectory(prefix='tidal-prefill-') as directory:
        root=Path(directory)
        for name,flags in [('dense',[]),('two-blocks',['--layers',8]),('ple',['--ple-layers','0,2']),
                           ('moe-ple',['--experts',4,'--experts-used',2,'--ple-layers','0,2'])]:
            model=root/(name+'.gguf');run(b/'tidal-dummy','--output',model,'--context',64,'--type','f32',*flags)
            run(b/'tidal-prefill-test',model,a.backend)
            trajectory(b/'tidal-cli',model,root/(name+'.json'),a.backend,Reference(model,a.backend),.004 if a.backend=='cuda' else .0004)
            print(name+': independent trajectory, layer order, state, pressure and recovery passed',flush=True)
        for tied in [True,False]:
            model=root/f'qwen35-{tied}.gguf';write_gguf(model,*fixture(tied=tied,explicit=True,mtp=True))
            run(b/'tidal-prefill-test',model,a.backend)
            trajectory(b/'tidal-cli',model,root/f'qwen35-{tied}.json',a.backend,Qwen35Reference(model,a.backend),.004 if a.backend=='cuda' else .0004)
            print(f'qwen35 tied={tied}: independent trajectory and state checks passed',flush=True)
        for kind in ['f32','f16','bf16','q8_0','q4_0']:
            model=root/(kind+'.gguf');run(b/'tidal-dummy','--output',model,'--context',64,'--type',kind)
            batch_matrices(b/'tidal-matvec-test',model,a.backend)
            trajectory(b/'tidal-cli',model,root/(kind+'.json'),a.backend,Reference(model,a.backend),.025 if kind.startswith('q') else .004 if a.backend=='cuda' else .012)
            print(kind+': isolated batch matrices and full prompt/continuation oracle passed',flush=True)
    print('Layer-major prefill validation passed')


if __name__=='__main__': main()
