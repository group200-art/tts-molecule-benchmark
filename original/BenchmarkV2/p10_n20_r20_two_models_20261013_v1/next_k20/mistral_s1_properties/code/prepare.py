"""Collect fixed sampled-R20 terminals; reuse exact original property requests."""
import argparse,math
from collections import Counter,defaultdict
from common import *

def main(check_current=False):
    require_allocation();config=verify_dependencies();execution=read(OUT/'config/execution.json')
    manifests=read(OUT/'config/manifests.json');bytask={};byuid={};sources={}
    def tracked(path):
        p=Path(path);blob=p.read_bytes();h=hashlib.sha256(blob).hexdigest()
        if str(p)in sources:assert sources[str(p)]==h
        sources[str(p)]=h
        return json.loads(gzip.decompress(blob) if p.suffix=='.gz' else blob)
    for task,spec in manifests.items():
        assert sha(spec['path'])==spec['sha256']
        rows=[json.loads(l) for l in Path(spec['path']).read_text().splitlines() if l.strip()]
        bytask[task]=rows
        for r in rows:byuid[task,r['input_uid']]=r
    all_candidates=[];original_filled=[];missing_generation=[];counts=Counter()
    newroot=Path(execution['new_generation_root'])
    def collect(terminal,record,path,hash_value):
        task,uid=record['task_id'],record['input_uid'];known=byuid[task,uid]
        assert record==known
        assert terminal['model_id']==execution['model_id'] and terminal['task_id']==task and terminal['input_uid']==uid
        assert terminal['source_smiles']==known['source_smiles'] and 0<=terminal['original_slot_id']<20
        assert terminal['method_id']==execution['method_id'] and terminal['input_idx']==known['input_idx']
        keys=['candidate_id','model_id','method_id','task_id','input_uid','input_idx','original_slot_id','source_smiles','raw_smiles','raw_text']
        c={k:terminal[k] for k in keys}
        c.update(generation_source_path=str(path),generation_source_sha256=hash_value,original_terminal_sha256=digest(terminal))
        all_candidates.append(c)
    if execution['kind']=='s1':
        oldroot=Path(execution['old_root']);units=tracked(oldroot/'units.json');plan=tracked(oldroot/'plan.json');plan_sha=sha(oldroot/'plan.json')
        assert sha(newroot/'units.json')==sha(oldroot/'units.json')
        assert sha(newroot/'plan.json')==plan_sha
        oldproperties=tracked(ROOT/'BenchmarkV2/k20_alignment_20261002_v1/configs/properties.json')
        assert oldproperties['identity_digest']==config['identity_digest']
        assert all(oldproperties['tasks'][t]['cache_contract']==config['tasks'][t]['cache_contract'] for t in TASKS)
        for unit in units:
            if unit['method']!='s1':continue
            records=bytask[unit['task_id']][unit['start']:unit['stop']];assert digest(records)==unit['records_sha256']
            for start in range(0,len(records),4):
                part=records[start:start+4];batchid=unit['unit_id']+'_b%02d'%(start//4)
                relative=Path('runs')/unit['unit_id']/batchid/'generation.json.gz';oldpath=oldroot/relative;newpath=newroot/relative
                path=oldpath if oldpath.exists() else newpath
                if not path.exists():missing_generation.append(str(path));counts['missing_generation_slots']+=20*len(part);continue
                if oldpath.exists() and newpath.exists():assert sha(oldpath)==sha(newpath),'Ambiguous generation artifacts'
                g=tracked(path);assert g['status']=='complete' and g['plan_sha256']==plan_sha
                assert g['batch']['batch_id']==batchid and g['batch']['records']==part and g['batch']['method']=='s1'
                assert g['feedback_batches']==0 and len(g['lanes'])==20*len(part)
                expected_lanes=[dict(lane_id=r['input_uid']+'_r%02d'%i,record=r,repeat=i) for r in part for i in range(20)]
                assert g['batch']['lanes']==expected_lanes
                assert [{k:l[k] for k in ['lane_id','record','repeat']} for l in g['lanes']]==expected_lanes
                terminals=[]
                for lane in g['lanes']:
                    t=lane['terminal'];assert lane['repeat']==t['original_slot_id']
                    assert t['execution']=='BV2-REMAINING-THREE-K20-v1'
                    ident={k:t[k] for k in ['method_id','model_id','task_id','input_uid','original_slot_id','execution']}
                    assert t['candidate_id']=='k20-terminal-'+digest(ident)
                    collect(t,lane['record'],path,sources[str(path)]);terminals.append(t)
                for r in part:assert [t['original_slot_id'] for t in terminals if t['input_uid']==r['input_uid']]==list(range(20))
                counts['generated_slots']+=len(terminals);counts['old_generated_slots' if path==oldpath else 'new_generated_slots']+=len(terminals)
                complete=oldpath.parent/'terminals/complete.json'
                if complete.exists():
                    receipt=tracked(complete);assert receipt['status']=='complete' and receipt['plan_sha256']==plan_sha
                    reqpath=complete.parent/'requests.json';request=tracked(reqpath)
                    assert receipt['requests_sha256']==sha(reqpath)
                    assert request['candidates']==terminals and request['records']==part
                    assert request['cache_contract']==config['tasks'][unit['task_id']]['cache_contract']
                    resultpath=Path(receipt['result_path']);result=tracked(resultpath)
                    assert result['requests_sha256']==sha(reqpath) and len(result['filled'])==len(terminals)
                    for t,c in zip(terminals,result['filled']):
                        assert all(c[k]==v for k,v in t.items()),'Cached properties changed original terminal'
                        compact_keys=['input_uid','raw_smiles','source_smiles','request_id','rdkit_valid','canonical_representations','raw_properties','properties','chemistry_basis','property_error','missing_reason','property_response_missing_reason']
                        original_filled.append((unit['task_id'],{k:c[k] for k in compact_keys if k in c},resultpath))
                    counts['original_evaluated_slots']+=len(result['filled'])
    elif execution['kind']=='icl':
        units=tracked(newroot/'units.json');contract=tracked(newroot/'contract.json');contract_sha=sha(newroot/'contract.json')
        assert contract['model']==execution['model_id'] and contract['visible_outputs']==20 and contract['inputs']==3905
        gate=tracked(newroot/'engineering_gate.json');runner=newroot.parent/'llama_icl_sampled.py'
        assert gate['status']=='PASS' and gate['contract_sha256']==contract_sha and gate['runner_sha256']==sha(runner)
        sources[str(runner)]=sha(runner)
        for path,h in contract['immutable_sources'].items():assert sha(path)==h;sources[path]=h
        for item in gate['probe_files']:assert sha(item['path'])==item['sha256'];sources[item['path']]=item['sha256']
        prompts={t:{p['record']['input_uid']:p for p in tracked(contract['prompt_files'][t])} for t in TASKS}
        for unit in units:
            unitdir=newroot/'runs'/unit['unit_id'];part=bytask[unit['task_id']][unit['start']:unit['stop']]
            assert unit['n_inputs']==len(part)
            assert [u for batch in unit['batches'] for u in batch['input_uids']]==[r['input_uid'] for r in part]
            paths=[unitdir/(b['batch_id']+'.json.gz') for b in unit['batches']]
            assert set(unitdir.glob('*.json.gz')).issubset(set(paths)),'Unplanned generation file'
            receiptpath=unitdir/'complete.json';receipt=None
            if receiptpath.exists():
                receipt=tracked(receiptpath)
                assert receipt['status']=='GENERATION_COMPLETE_PROPERTIES_PENDING' and receipt['unit']==unit
                assert receipt['contract_sha256']==contract_sha and receipt['runner_sha256']==sha(runner)
                assert receipt['terminal_slots']==unit['n_inputs']*20
                assert receipt['paths']==[dict(path=str(p),sha256=sha(p)) for p in paths]
            else:missing_generation.append(str(receiptpath))
            for batch,path in zip(unit['batches'],paths):
                if not path.exists():
                    missing_generation.append(str(path));counts['missing_generation_slots']+=20*len(batch['input_uids']);continue
                g=tracked(path);assert g['status']=='complete' and not g['properties_evaluated']
                lane_keys=[[unit['task_id'],uid,i] for uid in batch['input_uids'] for i in range(20)]
                assert g['identity']==dict(execution=contract['execution_id'],contract_sha256=contract_sha,batch_id=batch['batch_id'],ordered_lane_keys=lane_keys)
                assert [[o['record']['task_id'],o['record']['input_uid'],o['repeat']] for o in g['outputs']]==lane_keys
                payload=json.dumps([contract['execution_id'],batch['batch_id'],lane_keys],ensure_ascii=False,separators=(',',':')).encode()
                assert g['seed']==int(hashlib.sha256(payload).hexdigest()[:15],16)%2147483647
                for key,value in contract['decoding'].items():assert g['effective_decoding'][key]==value
                for rowidx,o in enumerate(g['outputs']):
                    r=o['record'];repeat=o['repeat'];p=prompts[r['task_id']][r['input_uid']]
                    assert o['prompt_sha256']==p['prompt_sha256'] and o['demo_uids']==p['demo_uids']
                    assert o['input_ids'][0]==128000 and 1<=o['output_tokens']<=100
                    assert o['seed']==g['seed'] and o['batch_row']==rowidx and o['raw_smiles']==o['parse'][0]
                    identity=dict(model_id=execution['model_id'],method_id=execution['method_id'],task_id=r['task_id'],input_uid=r['input_uid'],repeat=repeat,generation_contract_sha256=contract_sha)
                    t=dict(model_id=execution['model_id'],method_id=execution['method_id'],task_id=r['task_id'],input_uid=r['input_uid'],input_idx=r['input_idx'],source_smiles=r['source_smiles'],original_slot_id=repeat,candidate_id='icl-r20-terminal-'+digest(identity),raw_smiles=o['raw_smiles'],raw_text=o['raw_text'])
                    collect(t,r,path,sources[str(path)]);counts['generated_slots']+=1
    else:raise AssertionError(execution['kind'])
    keys=[(c['task_id'],c['input_uid'],c['original_slot_id']) for c in all_candidates]
    assert len(keys)==len(set(keys)) and len({c['candidate_id'] for c in all_candidates})==len(keys)
    if check_current:
        write(OUT/'reports/current_input_check.json',dict(status='PASS_CURRENT_IMMUTABLE_INPUTS',complete_generation=not missing_generation,counts=dict(counts),missing_generation=missing_generation,source_hashes=sources,unknown_data_errors=0,properties_started=False))
        print(json.dumps(dict(status='PASS_CURRENT_IMMUTABLE_INPUTS',counts=dict(counts),missing_batches=len(missing_generation))),flush=True);return
    assert not missing_generation,'Incomplete generation blocks full property collection'
    assert len(all_candidates)==78100
    slots_by_input=defaultdict(set)
    for task,uid,slot in keys:slots_by_input[task,uid].add(slot)
    for task,records in bytask.items():
        for r in records:assert slots_by_input[task,r['input_uid']]==set(range(20))
    requests={};cache={};evidence={};diagnostics=Counter()
    for c in all_candidates:
        ident=dict(raw_smiles=c['raw_smiles'],source_smiles=c['source_smiles'],cache_contract=config['tasks'][c['task_id']]['cache_contract'])
        rid='prop-request-v1-'+digest(ident);c['request_id']=rid
        if rid in requests:assert requests[rid]['identity']==ident
        else:requests[rid]=dict(request_id=rid,identity=ident,task_id=c['task_id'])
    def accept(task,c,path):
        rid=c.get('request_id')
        if rid not in requests:return
        identity=requests[rid]['identity'];record=byuid[task,c['input_uid']]
        assert c['raw_smiles']==identity['raw_smiles'] and record['source_smiles']==identity['source_smiles']
        canonical=c.get('canonical_representations');raw=c.get('raw_properties');comparison=c.get('properties')
        if not canonical:return
        versions={v['environment']:v for v in canonical}
        if 'gellmo_inf'not in versions or 'gellmo_eval_py36'not in versions:return
        assert versions['gellmo_inf']['rdkit_version']=='2024.03.6' and versions['gellmo_eval_py36']['rdkit_version']=='2020.03.3'
        disagreement=versions['gellmo_inf']['valid']!=versions['gellmo_eval_py36']['valid']
        error=c.get('property_error');reason=c.get('missing_reason') or c.get('property_response_missing_reason')
        if raw is None and reason is None:reason='property_error:'+error['category'] if error else 'invalid_in_inf_or_eval'
        r=dict(request_id=rid,raw_smiles=c['raw_smiles'],source_smiles=record['source_smiles'],cache_contract=identity['cache_contract'],status='hit' if raw is not None else 'missing',rdkit_valid=c['rdkit_valid'],canonical_representations=canonical,chemistry_basis=c.get('chemistry_basis','gellmo_inf RDKit recheck; eval validity separately retained'),values=raw,missing_reason=reason,property_error=error,environment_validity_disagreement=disagreement,cached_comparison_properties=comparison)
        finite=isinstance(raw,dict) and all(isinstance(raw.get(p),(int,float)) and math.isfinite(raw[p]) for p in identity['cache_contract']['calculators'])
        if not finite and c['rdkit_valid'] and not known_failure(r):return
        if rid in cache:
            assert cache[rid]['cached_comparison_properties']==comparison,'Comparison cache conflict'
            for name in ['sim','sas']:
                if isinstance(raw,dict) and isinstance(cache[rid]['values'],dict):assert math.isclose(raw[name],cache[rid]['values'][name],abs_tol=1e-12,rel_tol=1e-12),'Raw quality cache conflict'
            if cache[rid]['values']!=raw:diagnostics['same_comparison_different_raw_occurrences']+=1
            return
        cache[rid]=r;evidence[rid]=dict(path=str(path),sha256=sources.get(str(path)) or sha(path))
        sources[str(path)]=evidence[rid]['sha256']
    for task,c,path in original_filled:accept(task,c,path)
    # Compatible current endpoints are optional cache suppliers, never alternate outputs.
    endpoint_audit=tracked(ROOT/'BenchmarkV2/naacl_long_paper_20261012_v1/contributions/tables/sources/endpoint_audit.json')
    for path,h in endpoint_audit['source_files'].items():
        assert sha(path)==h;sources[path]=h
        with gzip.open(path,'rt') as f:
            for line in f:
                r=json.loads(line);c=dict(r['selected_candidate'],input_uid=r['input_uid']);accept(r['task_id'],c,Path(path))
    p10props=ROOT/'BenchmarkV2/p10_n20_r20_two_models_20261013_v1/properties/results/winner_properties.jsonl.gz'
    if p10props.exists():
        sources[str(p10props)]=sha(p10props)
        with gzip.open(str(p10props),'rt') as f:
            for line in f:
                c=json.loads(line);accept(c['task_id'],c,p10props)
    reports=[]
    for task in TASKS:
        candidates=[c for c in all_candidates if c['task_id']==task]
        reqs=[dict(request_id=r['request_id'],identity=r['identity']) for r in requests.values() if r['task_id']==task]
        cached=[cache[r['request_id']] for r in reqs if r['request_id']in cache];missing=[r for r in reqs if r['request_id']not in cache]
        write(OUT/'prepared'/(task+'.json.gz'),dict(task_id=task,records=bytask[task],candidates=candidates,requests=reqs,cached_responses=cached,missing_requests=missing,cache_contract=config['tasks'][task]['cache_contract'],admet_columns=config['tasks'][task]['admet_columns'],evidence={r['request_id']:evidence[r['request_id']] for r in cached}))
        reports.append(dict(task_id=task,inputs=len(bytask[task]),slots=len(candidates),unique_requests=len(reqs),cache_hits=len(cached),new_requests=len(missing)))
    write(OUT/'reports/inventory.json',dict(status='READY_COMPLETE_GENERATION',execution=execution,inputs=3905,terminal_slots=78100,unique_requests=len(requests),cache_hits=len(cache),new_requests=len(requests)-len(cache),counts=dict(counts),diagnostics=dict(diagnostics),source_hashes=sources,tasks=reports))
    print(json.dumps(dict(status='READY_COMPLETE_GENERATION',unique_requests=len(requests),cache_hits=len(cache),new_requests=len(requests)-len(cache),tasks=reports)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check-current',action='store_true');a=p.parse_args();main(a.check_current)
