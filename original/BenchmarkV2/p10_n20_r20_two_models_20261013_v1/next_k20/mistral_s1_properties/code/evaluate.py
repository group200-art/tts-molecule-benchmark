"""Evaluate only missing fixed-winner properties, preserving original caches."""
import argparse,subprocess,time
from common import *

def child(role,inp,target,directory,admet=None):
    target=Path(target);directory=Path(directory)
    attempt=directory/'attempts'/(role+'_'+os.environ['SLURM_JOB_ID']);attempt.mkdir(parents=True,exist_ok=False)
    staged=attempt/('result.json.gz' if role=='normalize' else 'producer_output')
    args=[ENVS['gellmo_inf' if role=='admet' else 'gellmo_eval_py36'],'-B',str(ENGINE/'worker.py'),role,'--input',str(inp),'--output',str(staged)]
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
    execution=read(OUT/'config/execution.json');candidate_rows=[]
    for c in data['candidates']:
        r=mapping[c['request_id']]
        if 'cached_comparison_properties'in r:assert r['comparison_properties']==r['cached_comparison_properties'],'Normalization changed cached comparison values'
        row=dict(c,cache_contract=r['cache_contract'],rdkit_valid=r['rdkit_valid'],canonical_representations=r['canonical_representations'],chemistry_basis=r['chemistry_basis'],properties=r['comparison_properties'],raw_properties=r['raw_properties'],rounding_csv=r['rounding_csv'],rounding_versions=r['rounding_versions'],property_row_exists=r['raw_properties'] is not None,known_molecular_property_failure=r['known_molecular_property_failure'],property_error=r.get('property_error'),missing_reason=r.get('missing_reason'),environment_validity_disagreement=r.get('environment_validity_disagreement',False),property_cache_evidence=evidence[r['request_id']])
        assert row['model_id']==execution['model_id'] and row['method_id']==execution['method_id']
        candidate_rows.append(row)
    byuid={}
    for c in candidate_rows:byuid.setdefault(c['input_uid'],[]).append(c)
    rows=[]
    for record in data['records']:
        candidates=sorted(byuid.pop(record['input_uid']),key=lambda c:c['original_slot_id'])
        assert [c['original_slot_id'] for c in candidates]==list(range(20))
        assert all(c['source_smiles']==record['source_smiles'] and c['input_idx']==record['input_idx'] for c in candidates)
        rows.append(dict(model_id=execution['model_id'],method_id=execution['method_id'],task_id=task_id,input_uid=record['input_uid'],input_idx=record['input_idx'],source_smiles=record['source_smiles'],source_properties=record['source_properties'],candidates=candidates))
    assert not byuid and len(candidate_rows)==20*len(rows)
    output=OUT/'results/by_task'/(task_id+'.json.gz');write(output,rows)
    write(OUT/'reports'/(task_id+'.json'),dict(status='COMPLETE',task_id=task_id,inputs=len(rows),terminal_slots=len(candidate_rows),unique_requests=len(requests),reused_requests=len(data['cached_responses']),new_requests=len(missing),known_molecular_failure_slots=sum(c['known_molecular_property_failure'] for c in candidate_rows),invalid_slots=sum(not c['rdkit_valid'] for c in candidate_rows),unknown_property_failures=0,output_path=str(output),output_sha256=sha(output),input_path=str(path),input_sha256=sha(path)))
    print(json.dumps(read(OUT/'reports'/(task_id+'.json'))),flush=True)

def finalize():
    require_allocation();verify_dependencies();inventory=read(OUT/'reports/inventory.json');execution=read(OUT/'config/execution.json')
    assert inventory['status']=='READY_COMPLETE_GENERATION' and inventory['execution']==execution
    rows=[];seen=set();candidate_ids=set();taskreports=[]
    for task_id in TASKS:
        report=read(OUT/'reports'/(task_id+'.json'));assert report['status']=='COMPLETE' and report['unknown_property_failures']==0
        assert sha(report['output_path'])==report['output_sha256'] and sha(report['input_path'])==report['input_sha256']
        for r in read(report['output_path']):
            key=(r['task_id'],r['input_uid']);assert key not in seen;seen.add(key)
            assert r['model_id']==execution['model_id'] and r['method_id']==execution['method_id']
            assert [c['original_slot_id'] for c in r['candidates']]==list(range(20))
            for c in r['candidates']:
                assert c['candidate_id'] not in candidate_ids;candidate_ids.add(c['candidate_id'])
            rows.append(r)
        taskreports.append(report)
    assert len(rows)==3905 and len(candidate_ids)==78100
    for path,h in inventory['source_hashes'].items():assert sha(path)==h,'Immutable source changed: '+path
    target=OUT/'results/terminal_inputs.jsonl.gz';tmp=target.with_suffix('.tmp')
    assert not target.exists() and not tmp.exists();target.parent.mkdir(parents=True,exist_ok=True)
    with tmp.open('xb') as f:
        with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:
            for r in rows:z.write((json.dumps(r,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode())
    os.rename(str(tmp),str(target))
    write(OUT/'reports/final_audit.json',dict(status='PASS_COMPLETE_TERMINAL_PROPERTIES',model_id=execution['model_id'],method_id=execution['method_id'],inputs=len(rows),terminal_slots=len(candidate_ids),unique_requests=inventory['unique_requests'],reused_requests=inventory['cache_hits'],new_requests=inventory['new_requests'],unknown_property_failures=0,known_molecular_failure_slots=sum(r['known_molecular_failure_slots'] for r in taskreports),invalid_slots=sum(r['invalid_slots'] for r in taskreports),new_language_model_generation=0,gpu_used=False,source_files_unchanged=True,task_reports=taskreports,output_path=str(target),output_sha256=sha(target),inventory_sha256=sha(OUT/'reports/inventory.json'),code_sha256={str(p):sha(p) for p in ENGINE.glob('*.py')}))
    print(json.dumps(read(OUT/'reports/final_audit.json'),indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task');p.add_argument('--finalize',action='store_true');a=p.parse_args()
    if a.finalize:finalize()
    else:task(a.task or TASKS[int(os.environ['SLURM_ARRAY_TASK_ID'])])
