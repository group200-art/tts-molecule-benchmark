"""Only preselected native winners enter this explicit request/response boundary."""
import os
from pathlib import Path
import subprocess
import time
from contracts import ROOT,ENVS,read,write,sha,isolated_env
from property_bridge import requests_for


def request_batches(requests):
    return [requests[i:i+100] for i in range(0,len(requests),100)]


def checked_child(interpreter,script,arguments,directory,label):
    cmd=[interpreter,'-B',str(ROOT/'src'/script)]+list(map(str,arguments))
    event={'argv':cmd,'started':True,'completed':False,'returncode':None,'wall_seconds':None,'CUDA_VISIBLE_DEVICES':''}
    write(directory/(label+'_started.json'),event);start=time.perf_counter()
    with (directory/(label+'.stdout')).open('x') as out,(directory/(label+'.stderr')).open('x') as err:
        proc=subprocess.run(cmd,stdout=out,stderr=err,env=isolated_env(directory,cpu=True),cwd=str(directory))
    event.update(completed=proc.returncode==0,returncode=proc.returncode,wall_seconds=time.perf_counter()-start)
    write(directory/(label+'_process.json'),event)
    if proc.returncode:raise RuntimeError('Property/metric process failed: {} exit {}'.format(label,proc.returncode))
    return event


def bridge_winners(task,candidates,directory,prior_responses):
    spec=read(ROOT/'configs/properties.json')['tasks'][task]
    contract=spec['cache_contract'];requests,mapping=requests_for(candidates,contract)
    cache={r['request_id']:r for r in prior_responses}
    if len(cache)!=len(prior_responses):raise ValueError('Duplicate prior request')
    fresh=[r for r in requests if r['request_id'] not in cache]
    ledger={'candidate_count':len(candidates),'requested_unique':len(requests),'cache_hits':len(requests)-len(fresh),
            'new_requests':len(fresh),'cache_scope':'This RUN01 E only; exact request key','processes':[]}
    write(directory/'requests_and_mapping.json',{'requests':requests,'mapping':mapping,'new_request_ids':[r['request_id'] for r in fresh]})
    for idx,batch in enumerate(request_batches(fresh)):
        d=directory/'batches'/('batch{:03d}'.format(idx));d.mkdir(parents=True,exist_ok=False)
        inp=d/'requests.json';write(inp,{'task_id':task,'requests':batch,'admet_columns':spec['admet_columns']})
        ledger['processes'].append(checked_child(ENVS['gellmo_inf'],'p10_chemistry_worker.py',
                    ['--requests',inp,'--output',d/'chemistry'],d,'chemistry'))
        ledger['processes'].append(checked_child(ENVS['gellmo_eval_py36'],'p10_eval_worker.py',
                    ['--mode','produce','--input',inp,'--input-sha',sha(inp),'--chemistry',d/'chemistry/result.json','--output',d/'standard'],d,'standard'))
        responses=read(d/'standard/responses_raw.json')
        if len(responses)!=len(batch) or [r['request_id'] for r in responses]!=[r['request_id'] for r in batch]:raise ValueError('Response order/identity mismatch')
        for r in responses:
            if r['request_id'] in cache:raise ValueError('Duplicate cache computation')
            cache[r['request_id']]=r
    result={'requests':requests,'mapping':mapping,'responses':[cache[r['request_id']] for r in requests],
            'cache_contract':contract,'candidates':candidates}
    write(directory/'bridge.json',result);write(directory/'cache_raw.json',list(cache.values()));write(directory/'property_costs.json',ledger)
    return result
