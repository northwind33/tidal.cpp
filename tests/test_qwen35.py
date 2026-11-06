#!/usr/bin/env python3
"""Native qwen35 GGUF schema, independent equations, full-attention placement."""
import argparse
import json
import math
import struct
import tempfile
from pathlib import Path
from test_reference import run, PROMPT, np
from qwen35_reference import Qwen35Reference


def write_gguf(path, metadata, tensors):
    def string(s):
        b = s.encode(); return struct.pack('<Q', len(b)) + b
    def value(kind, v):
        if kind == 8: return string(v)
        if kind == 9:
            element, items = v
            return struct.pack('<IQ', element, len(items)) + b''.join(value(element, x) for x in items)
        return struct.pack({4:'<I', 5:'<i', 6:'<f', 7:'<?'}[kind], v)
    header = b'GGUF' + struct.pack('<IQQ', 3, len(tensors), len(metadata))
    for key, (kind, data) in metadata.items(): header += string(key) + struct.pack('<I', kind) + value(kind, data)
    offset = 0
    for name, data in tensors.items():
        shape = tuple(reversed(data.shape))
        header += string(name) + struct.pack('<I', len(shape)) + struct.pack('<'+'Q'*len(shape), *shape) + struct.pack('<IQ', 0, offset)
        offset += (data.nbytes+31)//32*32
    with path.open('wb') as out:
        out.write(header + b'\0' * (-len(header) % 32))
        for data in tensors.values():
            out.write(data.astype('<f4').tobytes()); out.write(b'\0' * (-data.nbytes % 32))


def fixture(tied=True, explicit=False, mtp=False):
    # Native file names and dimensions are specified independently of the C++ loader.
    c = dict(embedding_length=64, feed_forward_length=128, context_length=64, block_count=4+int(mtp),
             **{'attention.head_count':4, 'attention.head_count_kv':2, 'attention.key_length':64,
                'attention.value_length':64, 'ssm.conv_kernel':4, 'ssm.state_size':32,
                'ssm.group_count':2, 'ssm.time_step_rank':4, 'ssm.inner_size':128,
                'full_attention_interval':4, 'rope.dimension_count':16, 'nextn_predict_layers':int(mtp)})
    metadata = {'general.architecture':(8,'qwen35')}
    metadata.update({'qwen35.'+k:(4,v) for k,v in c.items()})
    metadata.update({'qwen35.rope.freq_base':(6,10000000.), 'qwen35.attention.layer_norm_rms_epsilon':(6,1e-6),
                     'qwen35.rope.dimension_sections':(9,(5,[3,3,2,0]))})
    mask = [True,False,True,False] if explicit else [True,True,True,False]
    if explicit: metadata['qwen35.attention.recurrent_layers'] = (9,(7,mask+[False]*int(mtp)))
    tensors = {}; rng = np.random.default_rng(42)
    def add(name, shape, kind='matrix'):
        r = rng.uniform(-1,1,shape).astype(np.float32)
        if kind == 'norm': r = 1 + .05*r
        elif kind == 'a': r = -np.exp(-.5+.2*r)
        elif kind == 'bias': r = -1+.2*r
        else: r *= .1 if kind == 'embedding' else .3 if kind == 'conv' else math.sqrt(3/shape[-1])
        tensors[name] = r
    add('token_embd.weight',(256,64),'embedding'); add('output_norm.weight',(64,),'norm')
    if not tied:add('output.weight',(256,64))
    for i,recurrent in enumerate(mask+[False]*int(mtp)):
        p=f'blk.{i}.'
        add(p+'attn_norm.weight',(64,),'norm'); add(p+'post_attention_norm.weight',(64,),'norm')
        add(p+'ffn_gate.weight',(128,64)); add(p+'ffn_up.weight',(128,64)); add(p+'ffn_down.weight',(64,128))
        if recurrent:
            add(p+'attn_qkv.weight',(256,64)); add(p+'attn_gate.weight',(128,64))
            add(p+'ssm_alpha.weight',(4,64)); add(p+'ssm_beta.weight',(4,64))
            add(p+'ssm_conv1d.weight',(256,4),'conv'); add(p+'ssm_a',(4,),'a'); add(p+'ssm_dt.bias',(4,),'bias')
            add(p+'ssm_norm.weight',(32,),'norm'); add(p+'ssm_out.weight',(64,128))
        else:
            add(p+'attn_q.weight',(512,64)); add(p+'attn_k.weight',(128,64)); add(p+'attn_v.weight',(128,64))
            add(p+'attn_output.weight',(64,256)); add(p+'attn_q_norm.weight',(64,),'norm'); add(p+'attn_k_norm.weight',(64,),'norm')
        if i==4:
            add(p+'nextn.eh_proj.weight',(64,128))
            for n in ['enorm','hnorm','shared_head_norm']:add(p+'nextn.'+n+'.weight',(64,),'norm')
    return metadata,tensors


def placement(data, c, count, backend, recurrent):
    assert data['architecture']=='qwen35' and data['attention_kind']=='full'
    assert data['attention_backend']=='CPU' and data['kv_type']=='BF16'
    assert data['backend'].startswith('CUDA' if backend=='cuda' else 'CPU')
    attn = sum(not x for x in recurrent); s=data['statistics']
    prefill=s.get('prefill_tokens',0) if backend=='cuda' else 0
    assert s['full_attention_calls']==attn*(count-prefill) and s['sparse_attention_calls']==0
    q=c['attention.head_count']*c['attention.key_length'];kv=c['attention.head_count_kv']*c['attention.key_length']
    down=4*count*(c['vocab']+attn*(q+2*kv)) if backend=='cuda' else 0
    up=4*count*(1+attn*(2+q)) if backend=='cuda' else 0
    if backend=='cuda': down += 12*len(data['generated'])
    if prefill:
        down=4*((count+1)*c['vocab']+1+attn*(count-prefill)*(q+2*kv)) + 4*prefill*kv*attn + 12*len(data['generated'])
        up=4*(count+attn*(2*count+(count-prefill)*q))
        assert s['prefill_kv_to_host_bytes']==4*prefill*kv*attn
    assert s['device_to_host_bytes']==down and s['host_to_device_bytes']==up, (s,down,up)
    for pos,step in enumerate(data['trace']):
        assert step['selections']==[dict(layer=i,tokens=list(range(pos+1))) for i,r in enumerate(recurrent) if not r]


def profile_placement(events, backend):
    nodes=[e for e in events if e['cat']=='node']
    assert nodes and all(e['args']['backend'].startswith('CUDA' if backend=='cuda' else 'CPU') for e in nodes)
    assert not any(e['args']['phase'].startswith('qsa') for e in events)
    for name in ['qk','softmax','weighted_v']:
        assert any(e['name']==name and e['args']['phase']=='full.attention' and e['args']['backend']=='CPU' for e in events)
    for phase in ['embedding','gdn','full.prepare','full.finish','head']:
        assert any(e['args']['phase']==phase for e in nodes)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bin-dir',type=Path,required=True)
    parser.add_argument('--backend',choices=['cpu','cuda'],default='cpu');args=parser.parse_args();b=args.bin_dir
    with tempfile.TemporaryDirectory(prefix='tidal-qwen35-') as directory:
        root=Path(directory)
        for label,kwargs in [('tied',{}),('mask-untied-mtp',dict(tied=False,explicit=True,mtp=True))]:
            metadata,tensors=fixture(**kwargs);model=root/(label+'.gguf');trace=root/(label+'.json')
            write_gguf(model,metadata,tensors)
            run(b/'tidal-cli','--model',model,'--backend',args.backend,'--context',32,'--tokens',','.join(map(str,PROMPT)),
                '--steps',4,'--repeat',2,'--json',trace)
            data=json.loads(trace.read_text());ref=Qwen35Reference(model,args.backend)
            placement(data,dict(ref.c,vocab=256),28,args.backend,ref.recurrent)
            assert data['model']['skipped_tensors']==(15 if kwargs.get('mtp') else 0)
            expected_params=sum(t.size for n,t in tensors.items() if not n.startswith('blk.4.'))
            assert data['model']['parameters']==expected_params
            for i,token in enumerate(PROMPT+data['generated']):
                expected,selected=ref.decode(token);actual=np.asarray(data['trace'][i]['logits'])
                tolerance=2e-3 if args.backend=='cuda' else 4e-4
                np.testing.assert_allclose(actual,expected,atol=tolerance,rtol=tolerance,err_msg=f'{label} token {i}')
                assert np.linalg.norm(actual-expected)/np.linalg.norm(expected)<.002
                if len(PROMPT)-1<=i<27:assert int(expected.argmax())==data['generated'][i-len(PROMPT)+1]
            run(b/'tidal-state-test',model,args.backend)
            print(f'{label}: independent logits, tiled GDN heads, full causal KV, transfer bytes and reset passed',flush=True)
        # EOS ends generation, but an EOS inside the prompt does not truncate it.
        eos_metadata = dict(metadata, **{'tokenizer.ggml.eos_token_id': (4, data['generated'][0])})
        eos_model = root/'eos.gguf'
        write_gguf(eos_model, eos_metadata, tensors)
        for stop, length in [(1, 1), (0, 4)]:
            run(b/'tidal-cli','--model',eos_model,'--backend',args.backend,'--context',32,
                '--tokens',','.join(map(str,PROMPT)),'--steps',4,'--repeat',2,'--stop-at-eos',stop,'--json',trace)
            result=json.loads(trace.read_text())
            assert result['generated']==data['generated'][:length]
            assert len(result['trace'])==len(PROMPT)+length
        run(b/'tidal-cli','--model',eos_model,'--backend',args.backend,'--stop-at-eos',2,ok=False)
        run(b/'tidal-cli','--model',eos_model,'--backend',args.backend,
            '--tokens',f"{data['generated'][0]},1",'--steps',0,'--json',trace)
        assert len(json.loads(trace.read_text())['trace'])==2
        profile=root/'profile.json'
        run(b/'tidal-bench','--model',model,'--backend',args.backend,'--context',32,'--prompt',3,'--steps',1,
            '--warmup',0,'--repetitions',1,'--profile','nodes','--json',profile)
        events=json.loads(profile.read_text())['profile']['events']
        profile_placement(events, args.backend)
        bad=root/'bad.gguf'
        mutations=[('qwen35.attention.value_length',(4,32)),('qwen35.ssm.inner_size',(4,127)),
                   ('qwen35.rope.dimension_sections',(9,(5,[3,3,3,0]))),('qwen35.block_count',(4,0)),
                   ('qwen35.rope.scaling.type',(8,'yarn')),('qwen35.rope.scaling.factor',(6,2.)),
                   ('tokenizer.ggml.eos_token_id',(4,256)),('tokenizer.ggml.eos_token_id',(5,-1)),
                   ('qwen35.attention.recurrent_layers',(9,(7,[True])))]
        for key,value in mutations:
            write_gguf(bad,dict(metadata,**{key:value}),tensors)
            run(b/'tidal-cli','--model',bad,'--backend',args.backend,ok=False)
        for name in ['blk.0.ssm_a','blk.1.attn_k.weight']:
            write_gguf(bad,metadata,{n:t for n,t in tensors.items() if n!=name})
            run(b/'tidal-cli','--model',bad,'--backend',args.backend,ok=False)
        write_gguf(bad,metadata,dict(tensors,**{'blk.4.private.weight':np.ones((32,),np.float32)}))
        run(b/'tidal-cli','--model',bad,'--backend',args.backend,ok=False)
        bad.write_bytes(model.read_bytes()[:-64]);run(b/'tidal-cli','--model',bad,'--backend',args.backend,ok=False)
        for size in [-1,65]:run(b/'tidal-cli','--model',model,'--backend',args.backend,'--context',size,ok=False)
    print('Native Qwen3.5 metadata, profiler and rejection checks passed')


if __name__=='__main__':main()
