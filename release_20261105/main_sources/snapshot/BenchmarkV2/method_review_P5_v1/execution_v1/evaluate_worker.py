"""One offline task/seed evaluation unit, with immutable upstream cache reuse."""
import os,sys,time,signal,traceback
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from contracts import ROOT, read, write, sha, manifest, task_contract, evaluation_directory, verify_inventory, scientific_identity
from runtime_common import ENVS, require_allocation, isolated_env, checked_child, slurm_environment, index_artifacts
from trajectory_evidence import collect
from self_refine import candidate, prefix_cost
from property_bridge import requests_for
from metrics_entry import aggregate_registered


def request_batches(requests):
    batches=[];batch=[];raws=set()
    for req in requests:
        raw=req['identity']['raw_smiles']
        if raw not in raws and len(raws)==100:
            batches.append(batch);batch=[];raws=set()
        batch.append(req);raws.add(raw)
    if batch:batches.append(batch)
    return batches


def execute(phase,task,seed,directory,costs):
    trajectories,missing,bad=collect(task,seed,pilot=phase=='E01',deep=True)
    if bad:raise ValueError('Saved trajectory audit failed: '+repr(bad[:3]))
    spec=read(ROOT/'configs/properties.json')['tasks'][task];contract=spec['cache_contract']
    previous=None
    if phase=='E02' and seed==51001 and task in ('BDP','BHMQ'):
        previous=evaluation_directory('E01',task,seed)
    if phase=='E03':previous=evaluation_directory('E02',task,seed)
    cache={};reused=[];origins={}
    if previous is not None and (previous/'result.json').exists():
        p=read(previous/'result.json')
        if p['scientific_identity_sha256']!=scientific_identity():raise ValueError('Evaluation reuse identity mismatch')
        reused=p['rows']
        if (previous/'cache.json').exists():
            cache=read(previous/'cache.json')
            origins={k:str(previous/'cache.json') for k in cache}
    if phase=='E02' and previous is not None and (len(reused)!=5 or read(previous/'result.json')['status']!='complete'):
        raise ValueError('Exact E01 evaluation reuse unavailable')
    reused_ids={x['candidate_id'] for x in reused}
    selected=[];trajectory_by_uid={};source_index=[]
    for entry in trajectories:
        t=entry['trajectory'];record=t['record'];trajectory_by_uid[record['input_uid']]=t
        source_index.append({k:v for k,v in entry.items() if k!='trajectory'})
        rounds=range(4) if phase=='E03' else [4]
        for r in rounds:
            c=candidate(record,seed,r,t['answers'][r])
            if c['candidate_id'] not in reused_ids:selected.append(c)
    costs['candidate_events_new']=len(selected)
    reqs,mapping=requests_for(selected,contract)
    costs.update(requests_unique=len(reqs),new_requests=0,cache_hit_request_events=0,endpoint_events=[])
    request_origins_before=dict(origins)
    for endpoint in sorted({c['original_slot_id'] for c in selected}):
        endpoint_candidates=[c for c in selected if c['original_slot_id']==endpoint]
        endpoint_requests,_=requests_for(endpoint_candidates,contract)
        new=[q for q in endpoint_requests if q['request_id'] not in cache]
        event=dict(r=endpoint,candidate_slots=len(endpoint_candidates),requested_unique=len(endpoint_requests),
                   new_requests=len(new),cache_hits=len(endpoint_requests)-len(new),batches=[])
        costs['endpoint_events'].append(event);costs['new_requests']+=len(new)
        costs['cache_hit_request_events']+=event['cache_hits']
        for idx,batch in enumerate(request_batches(new)):
            d=directory/'properties'/('r{}_batch{:03d}'.format(endpoint,idx));d.mkdir(parents=True,exist_ok=False)
            inp=d/'requests.json';write(inp,dict(task_id=task,requests=batch,admet_columns=spec['admet_columns']))
            first_process=len(costs['processes'])
            checked_child(ENVS['gellmo_inf'],'admet_worker.py',['--requests',inp,'--output',d/'admet'],d,'admet',costs['processes'])
            checked_child(ENVS['gellmo_eval_py36'],'eval_worker.py',['--mode','produce','--input',inp,'--input-sha',sha(inp),'--admet',d/'admet/admet_result.json','--output',d/'standard'],d,'standard',costs['processes'])
            responses=read(d/'standard/responses_raw.json')
            if len(responses)!=len(batch) or {r['request_id'] for r in responses}!={r['request_id'] for r in batch}:
                raise ValueError('Property response set mismatch')
            for response in responses:
                cache[response['request_id']]=response;origins[response['request_id']]=str(d/'standard/responses_raw.json')
            ai=read(d/'admet/admet_result.json')['identity']
            costs['admet_batches'].append(dict(ai,endpoint=endpoint,batch_path=str(d)))
            event['batches'].append(dict(path=str(d),request_ids=[q['request_id'] for q in batch],
                                         process_wall_seconds=sum(x['wall_seconds'] for x in costs['processes'][first_process:])))
    write(directory/'cache.json',cache,compact=True);write(directory/'cache_origins.json',origins)
    rows=list(reused)
    if selected:
        bridge=dict(requests=reqs,mapping=mapping,responses=[cache[q['request_id']] for q in reqs],cache_contract=contract)
        inp=directory/'slots_input.json';out=directory/'slots_result.json'
        write(inp,dict(records=manifest(task),candidates=selected,bridge=bridge),compact=True)
        checked_child(ENVS['gellmo_eval_py36'],'eval_worker.py',['--mode','slots','--input',inp,'--input-sha',sha(inp),'--output',out],directory,'slots',costs['processes'])
        rows+=read(out)['rows']
    for row in rows:
        t=trajectory_by_uid.get(row['input_uid'])
        if t is None:raise ValueError('Reused metric lacks identical saved trajectory')
        expected=candidate(t['record'],seed,row['r'],t['answers'][row['r']])
        if row['candidate_id']!=expected['candidate_id']:raise ValueError('Reused candidate identity mismatch')
        row['prefix_generation_cost']=prefix_cost(t,row['r'])
        if row['candidate_id'] not in reused_ids:
            rid=row['candidate']['request_id']
            row['offline_cost_reference']=dict(phase=phase,endpoint=row['r'],costs_path=str(directory/'costs.json'),
                response_origin=origins[rid],cached_from_prior_evaluation=rid in request_origins_before,
                independent_per_candidate_wall_seconds=None,
                basis='actual endpoint batch costs and request identity; shared batch time is not invented as per-molecule time')
    order={rec['input_uid']:rec['input_idx'] for rec in manifest(task)}
    rows.sort(key=lambda x:(x['r'],order[x['input_uid']]))
    expected_n=5 if phase=='E01' else len(manifest(task));rounds=range(5) if phase=='E03' else [4]
    coverage={};metrics={}
    for r in rounds:
        subset=[x['metric_row'] for x in rows if x['r']==r]
        complete=len(subset)==expected_n
        coverage[str(r)]=dict(expected_n=expected_n,actual_n=len(subset),complete=complete)
        metrics[str(r)]=dict(status='complete' if complete else 'partial',full_task=aggregate_registered(subset) if complete else None,
                             completed_subset=aggregate_registered(subset))
    complete=all(v['complete'] for v in coverage.values()) and not missing
    result=dict(phase=phase,task_id=task,rep_seed=seed,status='complete' if complete else 'partial',
                scientific_identity_sha256=scientific_identity(),coverage=coverage,metrics=metrics,rows=rows,
                missing_generation_uids=missing,reused_evaluation_path=str(previous) if previous else None,
                reused_candidate_ids=sorted(reused_ids),new_property_event_count=len(selected),
                property_cost_path=str(directory/'costs.json'),trajectory_source_index=source_index)
    write(directory/'result.json',result,compact=True)
    return result


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['E01','E02','E03'],required=True)
    p.add_argument('--task',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--gate',action='store_true')
    args=p.parse_args();require_allocation();verify_inventory()
    d=evaluation_directory(args.phase,args.task,args.seed);d.mkdir(parents=True,exist_ok=False)
    os.environ.update(isolated_env(d,cpu=True));os.chdir(str(d));start=time.perf_counter()
    status=dict(phase=args.phase,task_id=args.task,rep_seed=args.seed,status='started',slurm=slurm_environment(),started_at_unix=time.time())
    costs=dict(processes=[],admet_batches=[],failure=None,inference_model_calls=0)
    write(d/'started.json',status)
    def stop(signum,frame):raise RuntimeError('Offline allocation terminated; no retry')
    signal.signal(signal.SIGTERM,stop)
    try:
        result=execute(args.phase,args.task,args.seed,d,costs);status['status']=result['status']
    except BaseException as error:
        status['status']='failed';costs['failure']=repr(error)
        write(d/'exception.json',dict(exception=repr(error),traceback=traceback.format_exc()))
    finally:
        costs['worker_wall_seconds']=time.perf_counter()-start;status['finished_at_unix']=time.time()
        write(d/'costs.json',costs);write(d/'status.json',status)
        index_artifacts(d)
    if args.gate:
        if args.phase!='E01' or args.task!='BHMQ':raise PermissionError('Gate only in second pilot CPU unit')
        from gate import acceptance_and_continue
        acceptance_and_continue()
    if status['status']=='failed':raise SystemExit(1)


if __name__=='__main__':main()
