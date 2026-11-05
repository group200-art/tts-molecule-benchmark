"""Llama four-shot sampled R20, generation only; identical frozen demonstrations.

New explicit stochastic variant, not the existing deterministic beam4 K1 row.
First run the three fixed engineering probe batches; no score-dependent tuning.
"""
import argparse
import fcntl
import gzip
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
STAGE=HERE/'llama_icl_sampled'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    p=Path(p)
    with (gzip.open if p.suffix=='.gz' else open)(str(p),'rt') as f:return json.load(f)


def save(p,x):
    p=Path(p);assert STAGE in p.resolve().parents and not p.exists()
    p.parent.mkdir(parents=True,exist_ok=True)
    blob=(json.dumps(x,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n').encode()
    tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()))
    with tmp.open('xb') as f:
        if p.suffix=='.gz':
            with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(blob)
        else:f.write(blob)
    os.link(str(tmp),str(p));tmp.unlink()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--unit-index',type=int);ap.add_argument('--probe',action='store_true');ap.add_argument('--audit-only',action='store_true');args=ap.parse_args()
    contract=read(STAGE/'contract.json');manifest=read(STAGE/'units.json')
    for p,h in contract['immutable_sources'].items():assert sha(p)==h,p
    plans={};units=[]
    for task,p in contract['prompt_files'].items():
        rows=read(p)
        for row in rows:
            assert row['shots']==4 and hashlib.sha256(row['prompt'].encode()).hexdigest()==row['prompt_sha256']
            key=(task,row['record']['input_uid']);assert key not in plans;plans[key]=row
    assert len(plans)==3905
    if args.audit_only:
        assert sum(u['n_inputs'] for u in manifest)==3905
        print(json.dumps(dict(status='PASS_CPU_PROMPTS_AND_UNITS',inputs=3905,terminal_slots=78100)))
        return
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('NEXT_K20_EXECUTION')==contract['execution_id']
    assert os.environ.get('SLURM_RESTART_COUNT','0')=='0'
    if args.probe:units=[manifest[i] for i in contract['probe_unit_indices']]
    else:
        gate=read(STAGE/'engineering_gate.json');assert gate['status']=='PASS' and gate['contract_sha256']==sha(STAGE/'contract.json')
        assert gate['runner_sha256']==sha(__file__)
        assert args.unit_index is not None and 0<=args.unit_index<len(manifest)
        units=[manifest[args.unit_index]]
    attempt=STAGE/'attempts'/(os.environ['SLURM_JOB_ID']+'_'+str(os.getpid()));attempt.mkdir(parents=True,exist_ok=False)
    # Reuse only the already audited two-model model assembly, not its ToT
    # generation or sampling function. Redirect its publication root explicitly.
    code=REPO/'BenchmarkV2/k1_nine_methods_execution_20261011_v2/code'
    sys.path.insert(0,str(code));tp=importlib.import_module('tot_protocol');tp.ROOT=STAGE
    tb=importlib.import_module('tot_backend');backend=tb.Backend('llama',contract['model_assembly'],sha(STAGE/'contract.json'),attempt)
    backend.start()
    from transformers import GenerationConfig
    import torch
    parserpath=REPO/'BenchmarkV2/k20_alignment_20261002_v1/src'
    sys.path.insert(0,str(parserpath));from pplus_parser import parse_smiles
    for unit in units:
        directory=STAGE/'runs'/unit['unit_id'];directory.mkdir(parents=True,exist_ok=True)
        lock=(directory/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        batches=unit['batches'][:1] if args.probe else unit['batches']
        for batch in batches:
            path=directory/(batch['batch_id']+'.json.gz')
            ordered=[(plans[(unit['task_id'],uid)],r) for uid in batch['input_uids'] for r in range(20)]
            keys=[(unit['task_id'],p['record']['input_uid'],r) for p,r in ordered]
            ident=dict(execution=contract['execution_id'],contract_sha256=sha(STAGE/'contract.json'),batch_id=batch['batch_id'],ordered_lane_keys=keys)
            if path.exists():
                previous=read(path);assert previous['identity']==json.loads(json.dumps(ident)) and previous['status']=='complete';continue
            prompts=[p['prompt'] for p,r in ordered]
            encoded=[backend.tok.encode(p,add_special_tokens=True,truncation=False) for p in prompts];width=max(map(len,encoded));cap=100
            assert width+cap<=32768 and all(x[0]==128000 for x in encoded)
            ids=torch.tensor([[128009]*(width-len(x))+x for x in encoded],dtype=torch.long,device='cuda:0')
            mask=torch.tensor([[0]*(width-len(x))+[1]*len(x) for x in encoded],dtype=torch.long,device='cuda:0')
            cfg=GenerationConfig.from_dict(backend.defaults)
            update=dict(contract['decoding'],bos_token_id=128000,eos_token_id=128009,pad_token_id=128009,return_dict_in_generate=False)
            assert not cfg.update(**update);cfg.validate()
            payload=json.dumps([contract['execution_id'],batch['batch_id'],keys],ensure_ascii=False,separators=(',',':')).encode()
            seed=int(hashlib.sha256(payload).hexdigest()[:15],16)%2147483647;tb.seed_all(seed)
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
            with torch.inference_mode():seq=backend.model.generate(input_ids=ids,attention_mask=mask,generation_config=cfg,synced_gpus=False)
            torch.cuda.synchronize();elapsed=time.perf_counter()-started
            outputs=[]
            for i,(tokens,(p,r)) in enumerate(zip(seq.detach().cpu().tolist(),ordered)):
                assert tokens[:width]==ids[i].cpu().tolist();cont=tokens[width:];assert 1<=len(cont)<=100
                count=cont.index(128009)+1 if 128009 in cont else len(cont);assert all(t==128009 for t in cont[count:])
                text=backend.tok.decode(cont,skip_special_tokens=True,clean_up_tokenization_spaces=False);parsed=parse_smiles(text)
                outputs.append(dict(record=p['record'],repeat=r,raw_text=text,raw_smiles=parsed[0],parse=list(parsed),
                                    input_ids=encoded[i],generated_ids=cont[:count],input_tokens=len(encoded[i]),output_tokens=count,
                                    demo_uids=p['demo_uids'],prompt_sha256=p['prompt_sha256'],batch_row=i,seed=seed))
            assert len(outputs)==len(keys)==len(batch['input_uids'])*20
            save(path,dict(status='complete',identity=ident,outputs=outputs,effective_decoding=cfg.to_dict(),seed=seed,
                           model_generate_seconds=elapsed,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                           peak_reserved_bytes=torch.cuda.max_memory_reserved(),job=os.environ['SLURM_JOB_ID'],properties_evaluated=False))
            print(json.dumps(dict(batch=batch['batch_id'],terminals=len(outputs),generate_seconds=elapsed)),flush=True)
        if not args.probe:
            paths=[directory/(b['batch_id']+'.json.gz') for b in unit['batches']]
            complete=dict(status='GENERATION_COMPLETE_PROPERTIES_PENDING',unit=unit,
                          contract_sha256=sha(STAGE/'contract.json'),runner_sha256=sha(__file__),
                          paths=[dict(path=str(p),sha256=sha(p)) for p in paths],
                          terminal_slots=sum(len(read(p)['outputs']) for p in paths))
            assert complete['terminal_slots']==unit['n_inputs']*20
            if (directory/'complete.json').exists():assert read(directory/'complete.json')==complete
            else:save(directory/'complete.json',complete)
        lock.close()
    print('Selected generation batches complete; endpoint property evaluation remains separate.',flush=True)


if __name__=='__main__':main()
