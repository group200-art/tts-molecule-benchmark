"""Fail-closed engineering checks; never use task success to choose a variant."""
import argparse
import datetime
import gzip
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
OLD=HERE.parents[1]/'k20_remaining_three_v1'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    p=Path(p)
    with (gzip.open if p.suffix=='.gz' else open)(str(p),'rt') as f:return json.load(f)
def write(p,x):
    assert not p.exists()
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def s1(job):
    stage=HERE/'mistral_s1_resume';snapshot=read(HERE/'pre_smoke_source_snapshot.json')
    verified={}
    for p,h in snapshot['files'].items():
        path=Path(p)
        if OLD in path.parents or path.name in ['resume_mistral_s1.py','mistral_s1_smoke.sbatch','mistral_s1_full.sbatch']:
            assert sha(path)==h,p
            verified[p]=h
    unit=read(OLD/'units.json')[102]
    directory=stage/'runs'/unit['unit_id']
    attempts=list(directory.glob('attempt_'+job+'_*'));assert len(attempts)==1
    receipt=read(attempts[0]/'generation_receipt.json');assert receipt['unit']==unit
    assert len(receipt['generated_selected'])==1 and receipt['generated_selected'][0]['slots']==80
    record=receipt['generated_selected'][0];result=read(record['path']);assert sha(record['path'])==record['sha256']
    assert result['status']=='complete' and len(result['lanes'])==80 and len(result['batch']['records'])==4
    assert result['feedback_batches']==result['feedback_candidates']==0
    keys=[(r['record']['input_uid'],r['terminal']['original_slot_id']) for r in result['lanes']]
    assert len(set(keys))==80
    assert keys==[(r['input_uid'],i) for r in result['batch']['records'] for i in range(20)]
    assert result['plan_sha256']==sha(OLD/'plan.json')
    calls=[read(p) for p in Path(record['path']).parent.joinpath('calls').glob('*.json.gz')]
    for call in calls:
        assert call['status']=='complete' and call['plan_sha256']==sha(OLD/'plan.json')
        requests=[o['request'] for o in call['outputs']]
        payload=['BV2-REMAINING-THREE-K20-v1',result['batch']['batch_id'],[r['key'] for r in requests]]
        # Historical digest uses sorted keys and compact serialization.
        raw=json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        assert call['seed']==int(hashlib.sha256(raw).hexdigest()[:15],16)%2147483647
        d=call['effective_decoding']
        assert d['do_sample'] and d['temperature']==1 and d['top_p']==1 and d['top_k']==0 and d['num_beams']==1
    identity=read(attempts[0]/'model_identity.json')
    assert identity['model_id']=='gellmo_p6_mistral' and identity['method_main']=='s1-B512-sampled-R20'
    assert identity['versions']=={'torch':'2.7.0+cu128','transformers':'4.46.3','peft':'0.14.0'}
    assert identity['tokenizer']['bos']==1 and identity['tokenizer']['eos']==2
    for x in identity['parameter_placements']:
        assert x['device']=='cuda:0' and x['dtype']==('torch.float32' if x['kind']=='lora' else 'torch.bfloat16')
    return dict(status='PASS',job_id=job,original_plan_sha256=sha(OLD/'plan.json'),
                runner_sha256=sha(HERE/'resume_mistral_s1.py'),terminal_slots=80,unique_uid_slot_keys=80,
                exact_source_hashes_unchanged=verified,smoke_result=record,
                hardware=identity['gpu'],software=identity['versions'],model_identity=str(attempts[0]/'model_identity.json'),
                changed_hardware_disclosure='Continuation uses lab Blackwell; original history includes L40S. Same frozen model/software/batch/controller/seed rules; no bitwise equivalence claim across hardware.',
                smoke_reuse='Full continuation reads the exact saved80 terminal batch; no reroll.',
                property_evaluation_completed=False,success_rate_used_for_configuration_selection=False)

def icl(job):
    stage=HERE/'llama_icl_sampled';contract=read(stage/'contract.json');units=read(stage/'units.json')
    sources={p:sha(p) for p in contract['immutable_sources']};assert sources==contract['immutable_sources']
    attempts=list((stage/'attempts').glob(job+'_*'));assert len(attempts)==1
    model=read(attempts[0]/'model_identity.json');assert model['model_id']=='gellmo_p6_llama'
    assert model['contract_sha256']==sha(stage/'contract.json')
    assert model['tokenizer']['bos']==128000 and model['tokenizer']['eos']==128009
    assert model['versions']=={'torch':'2.7.0+cu128','transformers':'4.46.3','peft':'0.14.0'}
    timings=[];references=[];all_keys=[]
    for index in contract['probe_unit_indices']:
        unit=units[index];batch=unit['batches'][0]
        path=stage/'runs'/unit['unit_id']/(batch['batch_id']+'.json.gz');r=read(path)
        assert r['status']=='complete' and r['identity']['contract_sha256']==sha(stage/'contract.json')
        keys=[(unit['task_id'],u,k) for u in batch['input_uids'] for k in range(20)]
        assert r['identity']['ordered_lane_keys']==[list(x) for x in keys]
        assert [(o['record']['task_id'],o['record']['input_uid'],o['repeat']) for o in r['outputs']]==keys
        payload=json.dumps([contract['execution_id'],batch['batch_id'],keys],ensure_ascii=False,separators=(',',':')).encode()
        assert r['seed']==int(hashlib.sha256(payload).hexdigest()[:15],16)%2147483647
        prompts={p['record']['input_uid']:p for p in read(contract['prompt_files'][unit['task_id']])}
        for o in r['outputs']:
            p=prompts[o['record']['input_uid']]
            assert o['prompt_sha256']==p['prompt_sha256'] and o['demo_uids']==p['demo_uids']
            assert o['input_ids'][0]==128000 and 1<=o['output_tokens']<=100
            assert o['seed']==r['seed'] and o['raw_smiles']==o['parse'][0]
        for key,value in contract['decoding'].items():assert r['effective_decoding'][key]==value,(key,r['effective_decoding'][key],value)
        assert not r['properties_evaluated']
        timings.append(r['model_generate_seconds']);references.append(dict(path=str(path),sha256=sha(path)))
        all_keys.extend(keys)
    assert len(all_keys)==len(set(all_keys))==240
    return dict(status='PASS',job_id=job,contract_sha256=sha(stage/'contract.json'),runner_sha256=sha(HERE/'llama_icl_sampled.py'),
                terminal_slots=240,unique_lane_keys=240,source_sha256=sources,probe_files=references,
                hardware=model['gpu'],software=model['versions'],generation_seconds=sum(timings),
                batch_seconds=timings,linear_3905_input_generation_hours=sum(timings)/12*3905/3600,
                throughput_limits='Three engineering batches only; excludes loading/queue and properties; full-run task lengths differ.',
                current_beam4_k1_is_same_config=False,success_rate_used_for_configuration_selection=False,
                reuse_basis='Probe runs the same first full-unit batch IDs, exact ordered lanes, prompts, contract and RNG digest. Full runner checks identities and skips those files unchanged.',
                property_evaluation_completed=False)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('method',choices=['s1','icl']);ap.add_argument('--job',required=True);args=ap.parse_args()
    result=(s1 if args.method=='s1' else icl)(args.job)
    result['checked_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    folder='mistral_s1_resume' if args.method=='s1' else 'llama_icl_sampled'
    write(HERE/folder/'engineering_gate.json',result)
    print(json.dumps({k:v for k,v in result.items() if k in ['status','job_id','terminal_slots','generation_seconds','linear_3905_input_generation_hours']}))

if __name__=='__main__':main()
