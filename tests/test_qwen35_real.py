#!/usr/bin/env python3
"""Opt-in public 0.8B verification. Models/reports stay outside tracked sources."""
import argparse
import json
import os
import subprocess
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from qwen35_reference import Qwen35Reference
from test_qwen35 import placement, profile_placement


PROMPT = [248045,846,198,3710,369,220,17,478,220,17,30,17308,440,1066,279,1324,13,
          248046,198,248045,74455,198,248068,271,248069,271]
# Official tokenizer's single-turn, non-thinking "What is 2 + 2? Reply with just the number."


def execute(command, log):
    with log.open('w') as output:
        subprocess.run(list(map(str,command)),stdout=output,stderr=subprocess.STDOUT,check=True,timeout=900)


def isolated_matrices(ref, binary, model, log):
    """Compare every first-token matvec with identical inputs, independently.

    Whole Q8 trajectories can diverge when tiny reduction differences cross an
    activation's FP16 scale boundary. These local checks retain tight bounds.
    Expected outputs still come entirely from NumPy, never the native helper.
    """
    aliases = {'attn_qkv.weight':'gdn.qkv', 'attn_gate.weight':'gdn.z',
               'ssm_alpha.weight':'gdn.a', 'ssm_beta.weight':'gdn.b', 'ssm_out.weight':'gdn.out',
               'attn_q.weight':'attn.qg', 'attn_k.weight':'attn.k', 'attn_v.weight':'attn.v',
               'attn_output.weight':'attn.out', 'ffn_gate.weight':'ffn.gate',
               'ffn_up.weight':'ffn.up', 'ffn_down.weight':'ffn.down'}
    original = ref.mm
    count = 0
    with log.open('w') as stderr:
        process = subprocess.Popen([str(binary), str(model), 'cuda'], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=stderr, text=True)
        def compare(name, x):
            nonlocal count
            expected = original(name, x)
            parts = name.split('.', 2)
            alias = '.'.join(parts[:2])+'.'+aliases[parts[2]] if name.startswith('blk.') else name
            process.stdin.write(alias+' '+' '.join(format(float(v), '.9g') for v in x)+'\n')
            process.stdin.flush()
            line = process.stdout.readline()
            if not line: raise RuntimeError('Native matvec helper exited before returning a result')
            actual = np.fromstring(line.split(' ', 1)[1], sep=' ')
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5, err_msg=name)
            count += 1
            return expected
        ref.mm = compare
        try:
            ref.decode(PROMPT[0])
            process.stdin.close()
            assert process.wait(timeout=30)==0
        finally:
            del ref.mm  # Restore the class method without retaining a self-reference.
            ref.reset()
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
            process.stdin.close()
            process.stdout.close()
    return count


def main():
    p=argparse.ArgumentParser();p.add_argument('--bin-dir',type=Path,required=True)
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,default=Path('build/qwen35-validation/real'))
    p.add_argument('--llama-reference',type=Path,help='Optional separately built llama.cpp oracle helper')
    args=p.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    reports=[]
    for kind in ['BF16','Q8_0']:
        model=args.model_dir/f'Qwen3.5-0.8B-{kind}.gguf';trace=out/f'{kind}.trace.json'
        execute([args.bin_dir/'tidal-cli','--model',model,'--backend','cuda','--context',256,
                 '--tokens',','.join(map(str,PROMPT)),'--steps',16,'--repeat',2,'--json',trace],out/f'{kind}.log')
        data=json.loads(trace.read_text());tokens=PROMPT+data['generated']
        assert data['generated']==[19,248046], data['generated']  # "4", then the GGUF EOS token.
        assert data['model']['converted_tensors']==0 and data['model']['skipped_tensors']==15
        ref=Qwen35Reference(model,'cuda')
        assert ref.c['embedding_length']==1024 and ref.layers==24
        placement(data,dict(ref.c,vocab=248320),len(tokens),'cuda',ref.recurrent)
        matrices = isolated_matrices(ref, args.bin_dir/'tidal-matvec-test', model, out/f'{kind}-matvec.log')
        max_nrmse=max_abs=0.
        for i,token in enumerate(tokens):
            expected,selected=ref.decode(token);actual=np.asarray(data['trace'][i]['logits'])
            assert np.isfinite(actual).all() and selected==data['trace'][i]['selections']
            nrmse=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
            # Q8 activation scale/code boundaries amplify tiny differences over
            # 24 layers. Tight isolated checks above and all-token argmax below
            # complement this coarse trajectory guard; it is not BF16 accuracy.
            assert nrmse < (.005 if kind=='BF16' else .1), (kind,i,nrmse)
            assert int(actual.argmax())==int(expected.argmax()), (kind,i,'argmax')
            if len(PROMPT)-1<=i<len(tokens)-1:assert int(expected.argmax())==tokens[i+1],(kind,i,'greedy')
            max_nrmse=max(max_nrmse,nrmse);max_abs=max(max_abs,float(np.max(abs(actual-expected))))
        report=dict(format=kind,compared_tokens=len(tokens),max_nrmse=max_nrmse,max_abs=max_abs,
                    generated=data['generated'],placement=True,reset_replay=True,weights=data['model'],
                    isolated_matvecs=matrices)
        del ref
        profile=out/f'{kind}.profile.json'
        execute([args.bin_dir/'tidal-bench','--model',model,'--backend','cuda','--context',32,
                 '--tokens',','.join(map(str,PROMPT[:2])),'--steps',1,'--warmup',0,'--repetitions',1,
                 '--profile','nodes','--json',profile],out/f'{kind}-profile.log')
        profile_data=json.loads(profile.read_text())
        assert profile_data['profile']['dropped_events']==0
        profile_placement(profile_data['profile']['events'], 'cuda')
        report['node_placement']=True
        if args.llama_reference:
            llama=out/f'{kind}-llama.f32'
            execute([args.llama_reference,model,'trace',','.join(map(str,tokens)),llama],out/f'{kind}-llama.log')
            expected=np.fromfile(llama,dtype=np.float32).reshape(len(tokens),248320)
            # llama.cpp's CPU BF16 KV matmuls also round Q/probabilities. Use the
            # NumPy oracle above for precise F32 CPU-attention arithmetic; this
            # cross-engine check independently verifies the generated answer.
            assert all(int(expected[i].argmax())==tokens[i+1] for i in range(len(PROMPT)-1,len(tokens)-1))
            report['llama_greedy_agreement']=True
        reports.append(report)
        print(json.dumps(report),flush=True)
    (out/'report.json').write_text(json.dumps(reports,indent=2)+'\n')
    print('Public Qwen3.5-0.8B BF16/Q8_0 CUDA/CPU split verification passed')


if __name__=='__main__':main()
