"""Join fixed winners to exact raw/source/property-contract caches."""
import math
from collections import Counter,defaultdict
from common import *

def lines(path):
    with gzip.open(str(path),'rt') as f:
        for line in f:
            if line.strip():yield json.loads(line)

def main():
    require_allocation();config=verify_dependencies()
    manifests=read(OUT/'config/manifests.json');records={}
    for task,spec in manifests.items():
        assert sha(spec['path'])==spec['sha256']
        for line in Path(spec['path']).read_text().splitlines():
            r=json.loads(line);records[task,r['input_uid']]=r
    for base in [P10,LLAMA,ROOT/'BenchmarkV2/next3_ten_tasks_run01_v1']:
        old=read(base/'configs/properties.json')
        assert old['identity_digest']==config['identity_digest'] and old['identity_files']==config['identity_files']
        assert all(old['tasks'][t]['cache_contract']==config['tasks'][t]['cache_contract'] for t in TASKS)
    winners={};requests={};bytask=defaultdict(list);winner_sources={};counts=Counter()
    for model in ['mistral','llama']:
        path=OUT.parent/'voting'/('grouped_inputs_'+model+'.jsonl.gz');winner_sources[str(path)]=sha(path)
        seen=set()
        for row in lines(path):
            task,uid=row['task_id'],row['input_uid'];assert (task,uid) not in seen;seen.add((task,uid))
            src=records[task,uid];assert row['source_smiles']==src['source_smiles'] and row['source_properties']==src['source_properties']
            assert row['model_id']=='gellmo_p6_'+model and len(row['winners'])==20
            assert [w['round_idx'] for w in row['winners']]==list(range(20))
            for w in row['winners']:
                raw=w['raw_smiles'];key=(row['model_id'],task,uid,raw)
                identity=dict(raw_smiles=raw,source_smiles=src['source_smiles'],cache_contract=config['tasks'][task]['cache_contract'])
                rid='prop-request-v1-'+digest(identity)
                if rid in requests:assert requests[rid]['identity']==identity
                else:requests[rid]=dict(request_id=rid,identity=identity,task_id=task)
                item=dict(model_id=row['model_id'],task_id=task,input_uid=uid,input_idx=src['input_idx'],source_smiles=src['source_smiles'],source_properties=src['source_properties'],raw_smiles=raw,request_id=rid,winner_round_indices=[],winner_candidate_ids=[])
                previous=winners.setdefault(key,item);previous['winner_round_indices'].append(w['round_idx']);previous['winner_candidate_ids'].append(w['candidate_id']);counts['winner_instances']+=1
        assert len(seen)==3905
    assert counts['winner_instances']==156200
    cache={};evidence={};source_hashes={};conflicts=[];diagnostics=Counter()
    def source(path):
        path=Path(path)
        if str(path) not in source_hashes:source_hashes[str(path)]=sha(path)
        return dict(path=str(path),sha256=source_hashes[str(path)])
    def accept(c,task,uid,path):
        rid=c.get('request_id')
        if rid not in requests:return
        request=requests[rid];src=records[task,uid];identity=request['identity']
        assert c['raw_smiles']==identity['raw_smiles'] and src['source_smiles']==identity['source_smiles']
        assert config['tasks'][task]['cache_contract']==identity['cache_contract']
        if 'source_smiles'in c:assert c['source_smiles']==src['source_smiles']
        raw=c.get('raw_properties');comparison=c.get('properties');canonical=c.get('canonical_representations')
        if not canonical or len(canonical)!=2:return
        versions={x['environment']:x for x in canonical}
        if not all(x in versions for x in ['gellmo_inf','gellmo_eval_py36']):return
        assert versions['gellmo_inf']['rdkit_version']=='2024.03.6' and versions['gellmo_eval_py36']['rdkit_version']=='2020.03.3'
        finite=isinstance(raw,dict) and all(isinstance(raw.get(p),(int,float)) and math.isfinite(raw[p]) for p in identity['cache_contract']['calculators'])
        mismatch=versions['gellmo_inf']['valid']!=versions['gellmo_eval_py36']['valid']
        error=c.get('property_error');reason=c.get('property_response_missing_reason')
        if raw is None and reason is None:reason='property_error:'+error['category'] if error else 'invalid_in_inf_or_eval'
        response=dict(request_id=rid,raw_smiles=c['raw_smiles'],source_smiles=src['source_smiles'],cache_contract=identity['cache_contract'],status='hit' if raw is not None else 'missing',rdkit_valid=c['rdkit_valid'],chemistry_basis=c.get('chemistry_basis','gellmo_inf RDKit recheck; eval validity separately retained'),canonical_representations=canonical,environment_validity_disagreement=mismatch,values=raw,missing_reason=reason,property_error=error,cached_comparison_properties=comparison)
        if not finite and c['rdkit_valid'] and not known_failure(response):diagnostics['unusable_cached_valid_property_rows']+=1;return
        if rid in cache:
            if cache[rid]['values']!=raw:
                if cache[rid]['cached_comparison_properties']!=comparison:conflicts.append(dict(request_id=rid,previous=evidence[rid],new=source(path)))
                else:diagnostics['same_comparison_different_raw_cache']+=1
            return
        cache[rid]=response;evidence[rid]=source(path);diagnostics['reused_requests']+=1
    # Prioritize measurements tied to original voting, then compatible fixed caches.
    for task in TASKS:
        for phase in ['pilot','full']:
            path=P10/'cpu'/phase/task/'metrics.json';d=read(path);source(path)
            for budget in sorted(d['by_budget'],key=int):
                for row in d['by_budget'][budget]['inputs']:
                    for c in row['candidates']:accept(c,task,row['input_uid'],path)
    extra=ROOT/'BenchmarkV2/k20_nine_full_20261003_v1/p10_comparison/new_property_results.json'
    data=read(extra);assert data['property_identity_digest']==config['identity_digest']
    for row in data['results']:
        c=dict(row,properties=row['comparison_properties'],canonical_representations=[dict(environment='gellmo_inf',rdkit_version='2024.03.6',smiles=row['canonical_inf'],valid=row['rdkit_valid']),dict(environment='gellmo_eval_py36',rdkit_version='2020.03.3',smiles=row['canonical_eval'],valid=row['eval_rdkit_valid'])])
        accept(c,row['task_id'],row['input_uid'],extra)
    extra=ROOT/'BenchmarkV2/k20_nine_full_20261003_v1/p10_comparison/reused_property_evidence.json'
    for row in read(extra):accept(row['property_evidence'],row['task_id'],row['input_uid'],extra)
    endpoint_audit=read(ROOT/'BenchmarkV2/naacl_long_paper_20261012_v1/contributions/tables/sources/endpoint_audit.json')
    for path,h in endpoint_audit['source_files'].items():
        assert sha(path)==h
        for row in lines(path):accept(row['selected_candidate'],row['task_id'],row['input_uid'],path)
    sets=[(RECOVERY/'results/by_task',['ev','beam20','sampling40_prefix20','p6_tot20','s1','fewshot','self_refine','reflexion']),
          (LLAMA/'results/by_task',['ev','beam20','sampling40_prefix20','p6_tot20','s1','fewshot','self_refine','reflexion']),
          (ROOT/'BenchmarkV2/next3_ten_tasks_run01_v1/results/tasks',['beam20','sampling40_prefix20','p6_tot20'])]
    for directory,methods in sets:
      for task in TASKS:
       for method in methods:
        path=directory/(method+'__'+task+'.json')
        if not path.exists():continue
        d=read(path);source(path)
        for row in d['inputs']:
            assert row['source_properties']==records[task,row['input_uid']]['source_properties']
            for c in row['candidates']:accept(c,task,row['input_uid'],path)
       print(json.dumps(dict(event='cache_scan',task=task,directory=str(directory),reused=len(cache),needed=len(requests))),flush=True)
    write(OUT/'reports/cache_conflicts.json',conflicts);assert not conflicts,'Conflicting fixed comparison values require review'
    summary=[]
    for task in TASKS:
        reqs=[dict(request_id=r['request_id'],identity=r['identity']) for r in requests.values() if r['task_id']==task]
        rows=[v for k,v in winners.items() if k[1]==task]
        hits=[cache[r['request_id']] for r in reqs if r['request_id']in cache]
        missing=[r for r in reqs if r['request_id']not in cache]
        payload=dict(task_id=task,cache_contract=config['tasks'][task]['cache_contract'],admet_columns=config['tasks'][task]['admet_columns'],winner_keys=rows,requests=reqs,cached_responses=hits,missing_requests=missing,evidence={r['request_id']:evidence[r['request_id']] for r in hits})
        write(OUT/'prepared'/(task+'.json.gz'),payload)
        summary.append(dict(task_id=task,winner_keys=len(rows),winner_instances=sum(len(r['winner_round_indices']) for r in rows),unique_requests=len(reqs),cache_hits=len(hits),new_requests=len(missing)))
    write(OUT/'reports/inventory.json',dict(status='READY_FOR_CPU_PROPERTIES',winner_sources=winner_sources,source_hashes=source_hashes,identity_digest=config['identity_digest'],model_count=2,inputs=7810,winner_instances=counts['winner_instances'],unique_winner_keys=len(winners),unique_requests=len(requests),cache_hits=len(cache),new_requests=len(requests)-len(cache),tasks=summary,diagnostics=dict(diagnostics)))
    print(json.dumps(read(OUT/'reports/inventory.json'),indent=2),flush=True)

if __name__=='__main__':main()
