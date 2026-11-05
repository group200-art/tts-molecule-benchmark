"""Recheck complete saved candidate pools and recompute the deadline analyses."""
from deadline_scope import *
from collections import Counter
from itertools import zip_longest
import numpy as np
import stats as st


def main():
 require_allocation();started=time.perf_counter()
 st.OUT=RELEASE;st.write=save  # Only metadata destination changes; statistics are reused verbatim.
 records=manifest();expected={(r['task_id'],r['input_uid']):r for r in records}
 boot=st.Bootstrap(records,'offline_full_3905')
 assert boot.metadata['weight_sha256']==read(OUT/'baseline_statistics_reconciliation.json')['cluster_weights_sha256']
 baseline={(r['model'].lower(),r['method_id'],r['task_id'],r['input_uid']):r for r in lines(R/'results/per_input_quality_and_prefix.jsonl.gz')}
 stored={(r['analysis'],r['backbone'],r['method'],r['selector'] if r['selector']!='NA' else '',r['metric']):r for r in csv.DictReader((OUT/'offline_summary.csv').open())}
 report=dict(status='RUNNING',started_utc=now().isoformat(),configurations=[],source_files={},files={},missing_data=[],raw_response_missing_counts={},original_endpoint_comparisons=0,returned_candidate_positions=0,vote_proposal_positions=0)
 output=[];perinput=[];pertask=[];contrasts=[];availability=[]
 def source_check(path,expected_sha):
  if path not in report['source_files']:
   assert Path(path).is_file(), 'Missing candidate source: '+path
   actual=sha(path);assert actual==expected_sha,'Changed candidate source: '+path
   report['source_files'][path]=actual
  else:assert report['source_files'][path]==expected_sha
 def validate_records(bykey):
  assert set(bykey)==set(expected), 'Input denominator changed'
  return [bykey[r['task_id'],r['input_uid']] for r in records]
 def check_summary(analysis,b,m,selector,metric,est):
  old=stored[analysis,b,m,selector,metric]
  for k in ['estimate','ci_low','ci_high']:
   assert abs(est[k]-float(old[k]))<1e-10,(analysis,b,m,selector,metric,k,est,old)
 def summarize(analysis,b,m,selector,rows):
  ordered=validate_records(rows)
  for metric in ordered[0]:
   values=[r[metric] for r in ordered]
   if all(v is None for v in values):
    output.append(dict(analysis=analysis,backbone=b,method=m,selector=selector,metric=metric,estimate=None,ci_low=None,ci_high=None,n_inputs=3905));continue
   assert all(v is not None for v in values)
   a=np.asarray(values,float);est,_=boot.estimate(a);check_summary(analysis,b,m,selector,metric,est)
   output.append(dict(analysis=analysis,backbone=b,method=m,selector=selector,metric=metric,**est,n_inputs=3905))
   for task,idx in boot.tasks.items():pertask.append(dict(analysis=analysis,backbone=b,method=m,selector=selector,task=task,metric=metric,estimate=100*float(a[idx].mean()),n_inputs=len(idx)))
  for rec,vals in zip(records,ordered):perinput.append(dict(analysis=analysis,backbone=b,method=m,selector=selector,task=rec['task_id'],instance_id=rec['input_uid'],raw_source_id=rec['source_molecule_uid'],**vals))
 def diff(analysis,b,m,name,metric,a,z):
  contrasts.append(dict(analysis=analysis,backbone=b,method=m,contrast=name,metric=metric,**boot.difference(a,z),unit='percentage points',n_inputs=3905))
 def cached_label(rec,c):
  props=c['properties'];valid=bool(c['structural_valid']);finite=KERNEL.finite_values(props,list(rec['source_properties']))
  success=bool(valid and finite and KERNEL.strictly_improves(rec['source_properties'],props,list(rec['source_properties'])))
  delta=bool(success and KERNEL.meets_delta(rec['source_properties'],props,list(rec['source_properties']),{k:DELTAS[k] for k in rec['source_properties']}))
  assert c['success']==success and c['threshold_success']==delta
  return success
 for b in MODELS:
  for m in METHODS:
   p=OUT/'original'/('%s__%s.jsonl.gz'%(b,m));s=OUT/'offline_inputs'/('similarity__%s__%s.jsonl.gz'%(b,m))
   report['files'][str(p)]=sha(p);report['files'][str(s)]=sha(s)
   bykey={};miss_raw=0;success_positions=0
   for row,simrow in zip_longest(lines(p),lines(s)):
    assert row is not None and simrow is not None
    rec=row['record'];key=rec['task_id'],rec['input_uid'];assert rec==expected[key] and simrow['record']==rec and key not in bykey
    source_check(row['original_source_path'],row['original_source_sha256'])
    cs=row['candidates'];ss=simrow['candidate_similarity'];assert len(cs)==len(ss)==20
    assert [c['original_slot_id'] for c in cs]==list(range(20))
    for i,(c,v) in enumerate(zip(cs,ss)):
     assert isinstance(c['raw_smiles'],str);cached_label(rec,c)
     assert v['candidate_id']==c['candidate_id'] and v['original_position']==i and v['success']==c['success']
     if c['success']:
      assert isinstance(v['similarity_full_precision'],(int,float)) and math.isfinite(v['similarity_full_precision']) and 0<=v['similarity_full_precision']<=1
      success_positions+=1
     miss_raw+=int(c.get('raw_response') is None and c.get('raw_text') is None)
    vals={**{'SR@'+str(k):float(any(c['success'] for c in cs[:k])) for k in KS},'validity':sum(c['structural_valid'] for c in cs)/20,'SR_delta@20':float(any(c['threshold_success'] for c in cs))}
    for tau in [.2,.4,.6,.8]:
     for k in KS:vals['SR_sim>'+str(tau)+'@'+str(k)]=float(any(st.constrained(c['success'],v['similarity_full_precision'],tau) for c,v in zip(cs[:k],ss[:k])))
     vals['SR_delta_sim>'+str(tau)+'@20']=float(any(st.constrained(c['threshold_success'],v['similarity_full_precision'],tau) for c,v in zip(cs,ss)))
    assert vals==simrow['values']
    old=baseline[(b,m)+key]
    assert all(bool(vals['SR@'+str(k)])==old['SR_K'+str(k)] for k in KS)
    assert bool(vals['SR_delta@20'])==old['delta'] and sum(c['structural_valid'] for c in cs)==old['n_valid_slots']
    report['original_endpoint_comparisons']+=6;report['returned_candidate_positions']+=20;bykey[key]=vals
   ordered=validate_records(bykey);summarize('similarity',b,m,'',bykey)
   diff('similarity',b,m,'Sim > 0.6 minus unconstrained','SR@20',[v['SR_sim>0.6@20'] for v in ordered],[v['SR@20'] for v in ordered])
   availability.append(dict(backbone=b,method=m,analysis='similarity',inputs=3905,positions=78100,successful_positions_with_finite_similarity=success_positions,missing_raw_model_text=miss_raw,analysis_missing_records=0,status='PASS'))
   report['raw_response_missing_counts'][b+'__'+m]=miss_raw
   print('validated similarity',b,m,flush=True)
  selectors={k:{} for k in ['same20_score_order','same20_creation_order','same20_uniform_permutation']}
  cached={(r['record']['task_id'],r['record']['input_uid']):r for r in lines(OUT/'offline_inputs'/('got_ordering__'+b+'.jsonl.gz'))}
  for row in lines(OUT/'original'/(b+'__got.jsonl.gz')):
   rec=row['record'];key=rec['task_id'],rec['input_uid'];cs=row['candidates'];pool=row['pool']
   assert len(cs)==20 and len(pool)==28 and len({c['candidate_id'] for c in cs})==20
   assert set(c['candidate_id'] for c in cs)<=set(c['candidate_id'] for c in pool)
   creation=sorted(cs,key=lambda c:c['created_order']);ns=sum(c['success'] for c in cs)
   for name,seq in [('same20_score_order',cs),('same20_creation_order',creation)]:
    selectors[name][key]={**{'SR@'+str(k):float(any(c['success'] for c in seq[:k])) for k in KS},'validity':sum(c['structural_valid'] for c in seq)/20}
   selectors['same20_uniform_permutation'][key]={**{'SR@'+str(k):st.permutation_success(20,ns,k) for k in KS},'validity':sum(c['structural_valid'] for c in cs)/20}
   for name in selectors:
    assert selectors[name][key]==cached[key]['selectors'][name]
    assert selectors[name][key]['SR@20']==selectors['same20_score_order'][key]['SR@20'] and selectors[name][key]['validity']==selectors['same20_score_order'][key]['validity']
  for name,rows in selectors.items():summarize('got_ordering',b,'got',name,rows)
  for name in ['same20_creation_order','same20_uniform_permutation']:
   aa=validate_records(selectors[name]);zz=validate_records(selectors['same20_score_order'])
   for k in KS:diff('got_ordering',b,'got',name+' minus same20_score_order','SR@'+str(k),[v['SR@'+str(k)] for v in aa],[v['SR@'+str(k)] for v in zz])
  availability.append(dict(backbone=b,method='got',analysis='same20_ordering',inputs=3905,positions=78100,analysis_missing_records=0,status='PASS'))
  print('validated GoT same20',b,flush=True)
  vote_names=['Vote','First proposal per group','Uniform position per group','Diagnostic group success upper bound'];vsel={name:{} for name in vote_names}
  selector=extracted_function(V/'voting/extract.py',{'count_vote'},dict(Counter=Counter))['count_vote'];validity_disagreements=0
  for row in lines(OUT/'original'/('vote_pool_'+b+'.jsonl.gz')):
   rec=row['record'];key=rec['task_id'],rec['input_uid'];assert rec==expected[key]
   path=OUT/'vote_evaluation'/b/rec['task_id']/'inputs'/(rec['input_uid']+'.json');ann=read(path)
   assert ann['record']==rec and len(row['proposals'])==len(ann['proposals'])==400
   source_check(row['original_source_path'],row['original_source_sha256']);report['files'][str(path)]=sha(path)
   ps=row['proposals'];acs=ann['proposals'];assert [p['global_slot'] for p in ps]==list(range(400))
   for i,(p,c) in enumerate(zip(ps,acs)):
    assert c['original_proposal']==p and p['block']==i//20 and p['block_slot']==i%20
    assert isinstance(p['raw_response'],str) and isinstance(p['parsed_candidate'],str)
    cached_label(rec,c);validity_disagreements+=int(c['structural_valid']!=p['rdkit_valid_ev'])
   winners=[]
   for g in range(20):
    w,_=selector(ps[20*g:20*(g+1)]);assert w['candidate_id']==row['winners'][g]['candidate_id'];winners.append(w['global_slot'])
   group_s=[sum(c['success'] for c in acs[20*g:20*(g+1)]) for g in range(20)]
   assert group_s==ann['group_success_counts']
   for name,positions in [('Vote',winners),('First proposal per group',list(range(0,400,20)))]:
    vsel[name][key]={**{'SR@'+str(k):float(any(acs[j]['success'] for j in positions[:k])) for k in KS},'validity':sum(acs[j]['structural_valid'] for j in positions)/20}
   vsel[vote_names[2]][key]={**{'SR@'+str(k):st.vote_uniform(group_s,k) for k in KS},'validity':sum(c['structural_valid'] for c in acs)/400}
   vsel[vote_names[3]][key]={**{'SR@'+str(k):float(any(group_s[:k])) for k in KS},'validity':None}
   for name in vote_names:assert vsel[name][key]==ann['selectors'][name]
   old=baseline[(b,'voting')+key]
   assert all(bool(vsel['Vote'][key]['SR@'+str(k)])==old['SR_K'+str(k)] for k in KS)
   report['vote_proposal_positions']+=400
  for name,rows in vsel.items():summarize('vote_fixed_pool',b,'voting',name,rows)
  z=validate_records(vsel['Vote'])
  for name in vote_names[1:]:
   a=validate_records(vsel[name])
   for k in KS:diff('vote_fixed_pool',b,'voting',name+' minus Vote','SR@'+str(k),[v['SR@'+str(k)] for v in a],[v['SR@'+str(k)] for v in z])
  availability.append(dict(backbone=b,method='voting',analysis='fixed_pool',inputs=3905,positions=1562000,selector_metric_validity_disagreements_preserved=validity_disagreements,analysis_missing_records=0,status='PASS'))
  print('validated Vote',b,flush=True)
 assert report['original_endpoint_comparisons']==421740 and report['vote_proposal_positions']==3124000
 csv_release(RELEASE/'offline_summary.csv',output);csv_release(RELEASE/'offline_per_input_metrics.csv',perinput);csv_release(RELEASE/'offline_per_task_metrics.csv',pertask);csv_release(RELEASE/'offline_paired_differences.csv',contrasts)
 csv_release(RELEASE/'candidate_availability.csv',availability)
 for path,h in read(OUT/'production_freeze.json')['scientific_assets'].items():assert sha(path)==h
 report.update(status='PASS',completed_utc=now().isoformat(),wall_seconds=time.perf_counter()-started,configurations=availability,bootstrap=boot.metadata,
  missing_model_text_note='Unavailable original raw model prose remains missing; original candidate SMILES and property labels suffice for these saved-output analyses. No prose was reconstructed.',
  property_precision='Original comparison fields; successful labels recomputed using the unchanged original metric kernel. Similarity uses the saved full-precision RDKit 2020.03.3 recomputation.',
  similarity_cache_audit='Every original candidate ID, position, success label, finite successful similarity and constrained prefix recomputed from the complete cache; original property/predictor code hashes unchanged. No new predictor run in this audit.')
 save(RELEASE/'offline_audit.json',report)
 print(json.dumps({k:report[k] for k in ['status','original_endpoint_comparisons','returned_candidate_positions','vote_proposal_positions','wall_seconds']}),flush=True)

if __name__=='__main__':
 try:main()
 except BaseException as e:
  save(RELEASE/'offline_audit_failure.json',dict(status='FAILED_REQUIRES_INSPECTION',error=repr(e),utc=now().isoformat()));raise
