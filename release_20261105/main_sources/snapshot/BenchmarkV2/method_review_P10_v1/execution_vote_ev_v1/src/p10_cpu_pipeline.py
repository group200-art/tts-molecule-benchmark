"""Single pilot gate or single task full CPU processing; never generates proposals."""
import argparse
import gzip
import json
import os
from pathlib import Path
import sys
import time
import traceback
from contracts import (ROOT,RUN_ID,TASKS,BUDGETS,ENVS,read,write,sha,verify,digest,manifest,seed_for,candidate_identity,
                       require_allocation,verify_code,isolated_env,destination)
from p10_vote import classify_ev,prefix_vote,native_projection,prefix_cost
from p10_property_batches import bridge_winners,checked_child


def gzread(path):
    with gzip.open(str(path),'rt',encoding='utf-8') as f:return json.load(f)


def source_unit(task,index):
    plan=read(ROOT/'configs/plan.json')
    units=plan['pilot'] if index<5 else plan['full']
    matches=[u for u in units if any(r['task_id']==task and r['input_idx']==index for r in u['inputs'])]
    if len(matches)!=1:raise ValueError('Missing or overlapping unit coverage')
    return matches[0]


def read_input(record,allow_partial=False):
    unit=source_unit(record['task_id'],record['input_idx']);d=ROOT/'runs'/unit['unit_id']
    folder=d/'inputs'/record['task_id']/('{:04d}'.format(record['input_idx']))
    if (folder/'complete.json').exists():
        complete=read(folder/'complete.json');calls=complete['calls']
        if complete['input_uid']!=record['input_uid'] or complete['proposals']!=400 or len(calls)!=20:raise ValueError('Incomplete input')
    elif allow_partial:
        calls=[]
        if (d/'calls.jsonl').exists():
            for line in (d/'calls.jsonl').read_text().splitlines():
                try:event=json.loads(line)
                except ValueError:break  # A killed append is retained as incomplete raw evidence.
                if event.get('input_uid')==record['input_uid'] and event.get('event')=='result' and event.get('completed'):
                    if event['block']!=len(calls):raise ValueError('Noncontiguous completed calls')
                    calls.append(event)
        if not calls:return [],[]
    else:raise ValueError('Missing pilot complete record')
    prompt=read(folder/'prompt_tokens.json');rows=[]
    from p10_parser import extract_candidate
    from p10_generation import token_record
    for b,call in enumerate(calls):
        if call['input_uid']!=record['input_uid'] or call['block']!=b or call['seed']!=seed_for(TASKS.index(record['task_id']),record['input_idx'],b) or call['completed'] is not True:
            raise ValueError('Call order/identity/seed mismatch')
        verify(call['proposals_path'],call['proposals_sha256']);verify(call['tokens_path'],call['tokens_sha256'])
        props=gzread(call['proposals_path']);returned=gzread(call['tokens_path'])
        if len(props)!=20 or len(returned['continuation_ids'])!=20 or returned['prompt_prefix_matches']!=[True]*20:raise ValueError('Physical batch shape mismatch')
        for s,(row,ids) in enumerate(zip(props,returned['continuation_ids'])):
            slot=20*b+s
            if row['global_slot']!=slot or any(row.get(k)!=v for k,v in candidate_identity(record,slot).items()):raise ValueError('Candidate lineage mismatch')
            expected=token_record(prompt['input_ids']+ids,prompt['input_ids']);expected.pop('continuation_ids')
            if any(row[k]!=v for k,v in expected.items()):raise ValueError('Token accounting mismatch')
            if row['raw_response']!=row['raw_decode_untrimmed'].strip() or extract_candidate(row['raw_response'])!={k:row[k] for k in ('parsed_candidate','parse_method','n_tag_matches')}:
                raise ValueError('Continuation/parser mismatch')
            if row['prompt_sha256']!=prompt['prompt_sha256'] or row['seed']!=call['seed']:raise ValueError('Raw prompt/seed identity mismatch')
            rows.append(row)
        if call['tokens_through_first_eos']!=sum(r['tokens_through_first_eos'] for r in props):raise ValueError('Call token sum mismatch')
    return rows,calls


def process_task(task,phase,directory):
    records=manifest(task);records=records[:5] if phase=='pilot' else records[5:]
    directory.mkdir(parents=True,exist_ok=False);write(directory/'records.json',records)
    output_instances=[];candidate_union={};selection_refs=[];identity_totals={'slots':0,'MolFromSmiles':0,'MolToSmiles':0,'wall_seconds':0.}
    start=time.perf_counter();vote_seconds=0.
    # Finalize every selected prefix before constructing any property request.
    for record in records:
        rows,calls=read_input(record,allow_partial=phase=='full');d=directory/'inputs'/('{:04d}'.format(record['input_idx']));d.mkdir(parents=True,exist_ok=False)
        if not rows:
            write(d/'not_available.json',{'input_uid':record['input_uid'],'reason':'no_completed_physical_prefix','budgets':list(BUDGETS)})
            continue
        for row in rows:
            ev=classify_ev(row['parsed_candidate'],row['raw_response']);row.update(ev)
            identity_totals['slots']+=1;identity_totals['wall_seconds']+=ev['identity_wall_seconds']
            for key,value in ev['identity_tool_calls'].items():identity_totals[key]+=value
        with gzip.open(str(d/'classified.json.gz'),'wt',encoding='utf-8') as f:json.dump(rows,f,ensure_ascii=False,separators=(',', ':'))
        t=time.perf_counter();winners=prefix_vote(rows);vote_seconds+=time.perf_counter()-t
        if phase=='pilot' and [w['N_used'] for w in winners]!=list(BUDGETS):raise ValueError('Four native pilot outputs required')
        for w in winners:
            n=w['N_used'];w.update(input_uid=record['input_uid'],task_id=task,input_idx=record['input_idx'],
                                   prefix_cost=prefix_cost(calls,n))
            if w['prefix_cost'] is None:raise ValueError('Incomplete prefix cost')
            projection=native_projection(w,record)
            candidate_union.setdefault(projection['candidate_id'],projection)
            selection_refs.append({'input_uid':record['input_uid'],'N_used':n,'candidate_id':projection['candidate_id']})
            output_instances.append(w)
        write(d/'winners.json',winners)
    write(directory/'winners_all.json',output_instances)
    write(directory/'identity_and_vote_costs.json',{'identity':identity_totals,'vote_wall_seconds':vote_seconds,
                                                  'read_audit_identity_vote_wall_seconds':time.perf_counter()-start})
    prior=[]
    if phase=='full':
        pilot=ROOT/'cpu/pilot'/task/'cache_raw.json'
        prior=read(pilot)
    bridge=bridge_winners(task,list(candidate_union.values()),directory,prior)
    payload=directory/'metric_input.json';write(payload,{'task_id':task,'records':records,'selections':selection_refs,'bridge':bridge})
    checked_child(ENVS['gellmo_eval_py36'],'p10_eval_worker.py',['--mode','metrics','--input',payload,'--input-sha',sha(payload),
                  '--output',directory/'metrics.json'],directory,'metrics')
    metrics=read(directory/'metrics.json')
    if any(len(metrics['by_budget'][str(n)]['inputs'])!=sum(s['N_used']==n for s in selection_refs) for n in BUDGETS):raise ValueError('Native metric denominator mismatch')
    missing={str(n):[r['input_uid'] for r in records if not any(s['input_uid']==r['input_uid'] and s['N_used']==n for s in selection_refs)] for n in BUDGETS}
    write(directory/'complete.json',{'task_id':task,'phase':phase,'input_count':len(records),'output_instances':len(output_instances),
           'proposals_classified':identity_totals['slots'],'status':'pass' if not any(missing.values()) else 'partial_generation',
           'missing_uids_by_budget':missing,'worker_wall_seconds':time.perf_counter()-start,
           'metrics_sha256':sha(directory/'metrics.json'),'winners_sha256':sha(directory/'winners_all.json')})


def audit_gpu_units(phase):
    units=read(ROOT/'configs/plan.json')[phase];report=[]
    for u in units:
        d=ROOT/'runs'/u['unit_id'];status=read(d/'status.json');cost=read(d/'costs.json');model=read(d/'model_identity.json')
        if status['unit']!=u or status['status']!='pass':raise ValueError('GPU unit failed/incomplete: '+u['unit_id'])
        if cost['generate_started']!=u['planned_calls'] or cost['generate_completed']!=u['planned_calls'] or cost['proposals']!=u['planned_proposals'] or cost['inputs_complete']!=u['n_inputs']:
            raise ValueError('Unit physical budget mismatch')
        if model['gpu']['name']!='NVIDIA H100 80GB HBM3' or model['generation_config_passed']!=read(ROOT/'configs/generation.json'):
            raise ValueError('Pilot model/config identity mismatch')
        report.append({'unit_id':u['unit_id'],'status':'pass','calls':cost['generate_completed'],'proposals':cost['proposals'],
                       'model_identity_sha256':sha(d/'model_identity.json'),'worker_wall_seconds':cost['worker_wall_seconds']})
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['pilot','full'],required=True);args=p.parse_args()
    require_allocation('gate' if args.phase=='pilot' else 'cpu_full');verify_code()
    phase=args.phase;d=ROOT/'cpu'/phase
    if phase=='pilot':tasks=TASKS;d.mkdir(parents=True,exist_ok=False)
    else:
        idx=int(os.environ['SLURM_ARRAY_TASK_ID']);tasks=[TASKS[idx]];d.mkdir(parents=True,exist_ok=True)
    marker=d/('gate' if phase=='pilot' else tasks[0]+'_worker');marker.mkdir(exist_ok=False)
    os.environ.update(isolated_env(marker,cpu=True));os.chdir(str(marker));start=time.perf_counter()
    try:
        unit_audit=audit_gpu_units(phase) if phase=='pilot' else None
        for task in tasks:process_task(task,phase,d/task)
        result={'run_id':RUN_ID,'phase':phase,'tasks':tasks,'engineering_pass':True,'gpu_units':unit_audit,
                'chemistry_performance_is_gate':False,'property_exceptions_masked':False,'worker_wall_seconds':time.perf_counter()-start,
                'actual_agent_gate_review_required':phase=='pilot'}
        if phase=='pilot':
            if sum(x['calls'] for x in unit_audit)!=1000 or sum(x['proposals'] for x in unit_audit)!=20000:raise ValueError('Pilot scope')
            sizes={str(x.relative_to(ROOT)):x.stat().st_size for x in ROOT.rglob('*') if x.is_file()}
            total=sum(sizes.values());raw=sum(v for k,v in sizes.items() if k.startswith('runs/'))
            result.update(storage_bytes_now=total,pilot_raw_bytes=raw,projected_full_raw_bytes=raw*3905/50,
                          storage_plan='Lossless gzip + single input prompt-token reference; all failure/raw evidence retained')
            if total>30*1024**3 or raw*3905/50>24*1024**3:raise ValueError('Pilot storage projection leaves insufficient 30GiB headroom')
        write(marker/'result.json',result)
    except BaseException as exc:
        write(marker/'failure.json',{'run_id':RUN_ID,'engineering_pass':False,'exception':repr(exc),'traceback':traceback.format_exc(),
                                     'full_expansion_allowed':False,'worker_wall_seconds':time.perf_counter()-start});raise


if __name__=='__main__':main()
