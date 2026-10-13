"""Evaluate only missing fixed-winner properties, preserving original caches."""
import argparse,subprocess,time
from common import *

def child(role,inp,target,directory,admet=None):
    target=Path(target);directory=Path(directory)
    attempt=directory/'attempts'/(role+'_'+os.environ['SLURM_JOB_ID']);attempt.mkdir(parents=True,exist_ok=False)
    staged=attempt/('result.json.gz' if role=='normalize' else 'producer_output')
    args=[ENVS['gellmo_inf' if role=='admet' else 'gellmo_eval_py36'],'-B',str(OUT/'code/worker.py'),role,'--input',str(inp),'--output',str(staged)]
    if admet is not None:args+=['--admet',str(admet)]
    started=time.perf_counter()
    with (attempt/'stdout').open('x') as stdout,(attempt/'stderr').open('x') as stderr:
        process=subprocess.run(args,stdout=stdout,stderr=stderr,cwd=str(OUT),env=isolated(attempt))
    write(attempt/'process.json',dict(argv=args,returncode=process.returncode,wall_seconds=time.perf_counter()-started,input_sha256=sha(inp)))
    assert process.returncode==0,'Property producer failed: '+str(attempt/'stderr')
    assert not target.exists();target.parent.mkdir(parents=True,exist_ok=True);os.rename(str(staged),str(target))

def task(task_id):
    require_allocation();verify_dependencies()
    path=OUT/'prepared'/(task_id+'.json.gz');data=read(path)
    requests={r['request_id']:r for r in data['requests']};responses={r['request_id']:r for r in data['cached_responses']};evidence=dict(data['evidence'])
    missing=data['missing_requests'];directory=OUT/'runs'/task_id
    for start in range(0,len(missing),100):
        block=directory/('block%03d'%(start//100));batch=missing[start:start+100]
        inp=block/'requests.json';write(inp,dict(task_id=task_id,requests=batch,admet_columns=data['admet_columns']))
        admet=block/'admet/admet_result.json';raw=block/'standard/responses_raw.json'
        if not admet.exists():child('admet',inp,admet.parent,block)
        assert read(admet)['requests_sha256']==sha(inp)
        if not raw.exists():child('standard',inp,raw.parent,block,admet)
        answer=read(raw);assert {r['request_id'] for r in answer}=={r['request_id'] for r in batch}
        for r in answer:
            assert r['request_id'] not in responses;responses[r['request_id']]=r;evidence[r['request_id']]=dict(path=str(raw),sha256=sha(raw),computed_now=True)
        write(block/'complete.json',dict(status='COMPLETE',input_sha256=sha(inp),admet_sha256=sha(admet),response_sha256=sha(raw)))
        print(json.dumps(dict(event='property_block_complete',task_id=task_id,completed=min(start+100,len(missing)),total=len(missing))),flush=True)
    assert set(responses)==set(requests)
    for rid,r in responses.items():
        identity=requests[rid]['identity'];assert all(r[k]==identity[k] for k in ['raw_smiles','source_smiles','cache_contract'])
    normalize=directory/'normalize_input.json.gz';write(normalize,dict(cache_contract=data['cache_contract'],responses=[responses[r['request_id']] for r in data['requests']]))
    normalized=directory/'normalized_responses.json.gz'
    if not normalized.exists():child('normalize',normalize,normalized,directory)
    normalized_rows=read(normalized);assert {r['request_id'] for r in normalized_rows}==set(requests)
    mapping={r['request_id']:r for r in normalized_rows}
    rows=[]
    for w in data['winner_keys']:
        r=mapping[w['request_id']]
        if 'cached_comparison_properties'in r:assert r['comparison_properties']==r['cached_comparison_properties'],'Normalization changed cached comparison values'
        row=dict(w,cache_contract=r['cache_contract'],rdkit_valid=r['rdkit_valid'],canonical_representations=r['canonical_representations'],chemistry_basis=r['chemistry_basis'],properties=r['comparison_properties'],raw_properties=r['raw_properties'],rounding_csv=r['rounding_csv'],rounding_versions=r['rounding_versions'],property_row_exists=r['raw_properties'] is not None,known_molecular_property_failure=r['known_molecular_property_failure'],property_error=r.get('property_error'),missing_reason=r.get('missing_reason'),environment_validity_disagreement=r.get('environment_validity_disagreement',False),property_cache_evidence=evidence[r['request_id']])
        rows.append(row)
    output=OUT/'results/by_task'/(task_id+'.json.gz');write(output,rows)
    write(OUT/'reports'/(task_id+'.json'),dict(status='COMPLETE',task_id=task_id,unique_winner_keys=len(rows),winner_instances=sum(len(r['winner_round_indices']) for r in rows),unique_requests=len(requests),reused_requests=len(data['cached_responses']),new_requests=len(missing),known_molecular_failure_keys=sum(r['known_molecular_property_failure'] for r in rows),unknown_property_failures=0,output_path=str(output),output_sha256=sha(output),input_path=str(path),input_sha256=sha(path)))
    print(json.dumps(read(OUT/'reports'/(task_id+'.json'))),flush=True)

def finalize():
    verify_dependencies();inventory=read(OUT/'reports/inventory.json');rows=[];seen=set();taskreports=[]
    for task_id in TASKS:
        report=read(OUT/'reports'/(task_id+'.json'));assert report['status']=='COMPLETE'
        assert sha(report['output_path'])==report['output_sha256'] and sha(report['input_path'])==report['input_sha256']
        for r in read(report['output_path']):
            key=tuple(r[k] for k in ['model_id','task_id','input_uid','raw_smiles']);assert key not in seen;seen.add(key);rows.append(r)
        taskreports.append(report)
    assert len(rows)==inventory['unique_winner_keys']
    assert sum(len(r['winner_round_indices']) for r in rows)==156200
    for path,h in inventory['winner_sources'].items():assert sha(path)==h
    for path,h in inventory['source_hashes'].items():assert sha(path)==h
    target=OUT/'results/winner_properties.jsonl.gz';tmp=target.with_suffix('.tmp')
    assert not target.exists() and not tmp.exists()
    with tmp.open('xb') as f:
        with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:
            for r in rows:z.write((json.dumps(r,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode())
    os.rename(str(tmp),str(target))
    write(OUT/'reports/final_audit.json',dict(status='PASS_ALL_FIXED_WINNER_PROPERTIES_COMPLETE',model_count=2,per_model_inputs=3905,winner_instances=156200,unique_winner_keys=len(rows),unique_requests=inventory['unique_requests'],reused_requests=inventory['cache_hits'],new_requests=inventory['new_requests'],unknown_property_failures=0,known_molecular_failure_keys=sum(r['known_molecular_property_failure'] for r in rows),invalid_keys=sum(not r['rdkit_valid'] for r in rows),new_language_model_generation=0,gpu_used=False,all_source_hashes_unchanged=True,task_reports=taskreports,output_path=str(target),output_sha256=sha(target),inventory_sha256=sha(OUT/'reports/inventory.json')))
    print(json.dumps(read(OUT/'reports/final_audit.json'),indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task');p.add_argument('--finalize',action='store_true');a=p.parse_args()
    if a.finalize:finalize()
    else:task(a.task or TASKS[int(os.environ['SLURM_ARRAY_TASK_ID'])])
