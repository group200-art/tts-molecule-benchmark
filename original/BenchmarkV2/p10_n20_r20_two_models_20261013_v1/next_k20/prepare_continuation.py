"""Prepare isolated configurations and bounded checkpoint manifests; no jobs."""
import datetime
import hashlib
import json
import shutil
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
B=REPO/'BenchmarkV2'
OLD=B/'k20_remaining_three_v1'
TASKS=['BDP','BDQ','BPQ','DPQ','BDPQ','MPQ','BDMQ','BHMQ','BMPQ','HMPQ']


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if p.exists():assert p.read_text()==payload,p
    else:p.write_text(payload)


def main():
    stage=HERE/'mistral_s1_resume';stage.mkdir(exist_ok=True)
    for name in ['plan.json','units.json','inventory.json','probe_gate.json']:
        dst=stage/name
        if dst.exists():assert sha(dst)==sha(OLD/name)
        else:shutil.copyfile(str(OLD/name),str(dst))
    assert read(OLD/'probe_gate.json')['status']=='PASS'
    plan=read(OLD/'plan.json');units=read(OLD/'units.json');missing=[];cpu_only=[];complete=[];generation_slots=evaluated_slots=0
    total_gen_s=total_wall_s=0.
    for u in units:
        if u['method']!='s1':continue
        d=OLD/'runs'/u['unit_id'];row=dict(u,required_slots=(u['stop']-u['start'])*20,generated_slots=0,evaluated_slots=0,missing_generation_batches=[],partial_call_sources=[])
        for start in range(u['start'],u['stop'],4):
            batch=u['unit_id']+'_b%02d'%((start-u['start'])//4);bd=d/batch;n=min(4,u['stop']-start)*20
            if (bd/'generation.json.gz').exists():row['generated_slots']+=n
            else:
                row['missing_generation_batches'].append(dict(batch_id=batch,terminal_slots=n))
                for p in sorted((bd/'calls').glob('*.json.gz')):row['partial_call_sources'].append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size))
            if (bd/'terminals/complete.json').exists():row['evaluated_slots']+=n
        generation_slots+=row['generated_slots'];evaluated_slots+=row['evaluated_slots']
        if (d/'complete.json').exists():
            c=read(d/'complete.json');assert c['unit']==u and c['status']=='complete';complete.append(row)
            total_gen_s+=c['physical_cuda_seconds'];total_wall_s+=c['wall_seconds']
        elif row['missing_generation_batches']:missing.append(row)
        else:cpu_only.append(row)
    n_complete=sum(r['stop']-r['start'] for r in complete)
    remain=78100-generation_slots
    inv=dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
      old_runs_are_read_only=True,max_concurrent_gpu_workers_initial=1,
      mistral_s1=dict(protocol=plan['extensions']['s1'],plan_sha256=sha(OLD/'plan.json'),
        complete_unit_inputs=n_complete,complete_units=len(complete),generated_terminal_slots=generation_slots,
        evaluated_terminal_slots=evaluated_slots,missing_generation_terminal_slots=remain,
        missing_generation_input_equivalent=remain/20.,missing_property_terminal_slots=78100-evaluated_slots,
        gpu_resume_units=missing,cpu_only_unfinished_units=cpu_only,
        measured_complete_units_generate_seconds=total_gen_s,measured_complete_units_worker_seconds=total_wall_s,
        generation_only_one_gpu_hours_linear=total_gen_s/n_complete*(remain/20.)/3600.,
        timing_scope='Observed completed Mistral sampled-R20 units, mostly L40S. Linear planning extrapolation; partial call reuse can reduce it. Excludes queue delay and unresolved terminal property exception.'),
      properties_unresolved='Unit66 known QED/RemoveHs KekulizeException is separate CPU evaluation work; never regenerate, silently discard, or autoassign failure.',
      no_nine_method_completion_claim=True)
    # Update inventory only before any new generation; subsequent runs must not
    # overwrite the original preparation snapshot.
    ip=HERE/'continuation_inventory.json'
    if not ip.exists():ip.write_text(json.dumps(inv,ensure_ascii=False,indent=2)+'\n')
    else:
        prev=read(ip);assert prev['mistral_s1']['plan_sha256']==inv['mistral_s1']['plan_sha256']

    ll=HERE/'llama_icl_sampled';ll.mkdir(exist_ok=True)
    assembly=read(B/'k1_nine_methods_execution_20261011_v2/config/tot_contract.json')
    promptfiles={t:str(B/'p6_llama_main_ten_tasks_v1/configs'/('pplus_prompts_'+t+'.json')) for t in TASKS}
    lu=[]
    for t in TASKS:
        rows=read(promptfiles[t]);assert all(r['shots']==4 for r in rows)
        for start in range(0,len(rows),80):
            stop=min(start+80,len(rows));uid='llama_icl_'+t+'_%04d_%04d'%(start,stop)
            batches=[]
            for j in range(start,stop,4):batches.append(dict(batch_id=uid+'_b%02d'%((j-start)//4),input_uids=[r['record']['input_uid'] for r in rows[j:min(j+4,stop)]]))
            lu.append(dict(index=len(lu),unit_id=uid,task_id=t,start=start,stop=stop,n_inputs=stop-start,batches=batches))
    probe=[next(u['index'] for u in lu if u['task_id']==t and u['start']==0) for t in ['BDP','BDPQ','BHMQ']]
    model_assembly={k:assembly[k] for k in ['models','tokenizer','generation_defaults_path']}
    srcpaths=list(promptfiles.values())+[str(B/'k1_nine_methods_execution_20261011_v2/config/tot_contract.json'),str(B/'k1_nine_methods_execution_20261011_v2/code/tot_backend.py'),str(B/'k1_nine_methods_execution_20261011_v2/code/tot_protocol.py'),str(B/'k20_alignment_20261002_v1/src/pplus_parser.py'),assembly['generation_defaults_path']]
    lc=dict(execution_id='GELLMO-LLAMA-ICL4-SAMPLED-R20-20261013-v1',label='4-shot ICL sampled R20',
            model='gellmo_p6_llama',shots=4,visible_outputs=20,inputs=3905,total_endpoints=78100,
            replaces_current_beam4_K1=False,current_K1_reusable_as_sampled_repeat=False,
            decoding=dict(do_sample=True,temperature=1.,top_p=1.,top_k=0,num_beams=1,num_return_sequences=1,min_new_tokens=0,max_new_tokens=100,repetition_penalty=1.,length_penalty=1.,no_repeat_ngram_size=0),
            physical_batch='Four consecutive manifest inputs x20 independent sampled rows; fresh row state. Last batch may be smaller.',
            seed='First15hex SHA256 compactJSON([execution_id,batch_id,ordered(task,uid,repeat) lane keys]) modulo2147483647; reset per physical call.',
            method_adaptation='Same frozen demonstration records, prompt, parser, cap100. Explicitly replaces deterministic beam4 with T1 sampling, as previously authorized for K20. No property-feedback selection.',
            generated_rows='Retain all20, including duplicate/invalid/empty; no refill or quality-dependent retries.',
            model_assembly=model_assembly,prompt_files=promptfiles,immutable_sources={p:sha(p) for p in srcpaths},
            probe_unit_indices=probe,probe_scope='First4 inputs of BDP, BDPQ, BHMQ, all20 terminals. Engineering/time/memory only. Reused if exact final contract unchanged.',
            full_run_gate='engineering_gate.json status PASS and matching contract hash; root GPU coordinator required',
            max_concurrent_gpu_workers=1,property_evaluation='Separate CPU step; generation-complete is not evaluation-complete.')
    write(ll/'contract.json',lc);write(ll/'units.json',lu)
    print(json.dumps(dict(mistral_s1_missing_slots=remain,mistral_s1_gpu_units=[u['index'] for u in missing],
      mistral_s1_linear_generation_hours=inv['mistral_s1']['generation_only_one_gpu_hours_linear'],llama_icl_units=len(lu),llama_icl_probe_units=probe),indent=2))


if __name__=='__main__':main()
