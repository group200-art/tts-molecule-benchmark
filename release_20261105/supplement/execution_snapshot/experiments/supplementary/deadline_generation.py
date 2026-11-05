"""Complete-scope GoT summaries, and an end-to-end timing gate before launch."""
from deadline_scope import *
from aggregate import measures,workload
from engineering_gate import validate_row
import stats as st
import numpy as np
import gzip
import argparse

def summarize(mode):
 require_allocation();started=time.perf_counter();cfg=read(RELEASE/'scope.json')
 assert cfg['subset_sha256']==sha(RELEASE/'subset_50_manifest.jsonl') and cfg['units_sha256']==sha(RELEASE/'units.json')
 for path,h in read(OUT/'production_freeze.json')['generation_code'].items():assert sha(path)==h
 pilot=mode=='pilot';target=RELEASE/('pilot_pipeline' if pilot else 'generation')
 target.mkdir(parents=True,exist_ok=True)
 expected=subset(profile='pilot') if pilot else list(lines(RELEASE/'subset_50_manifest.jsonl'))
 expected_keys={(r['task_id'],r['input_uid']) for r in expected}
 roots=[OUT/'generation'/('pilot_'+b+'_1729_all_got_0_2') for b in MODELS] if pilot else [unit_path(u) for u in read(RELEASE/'units.json')]
 missing=[];paths=[]
 for root in roots:
  receipt=root/'completion.json'
  if not receipt.exists():missing.append(str(root));continue
  assert read(receipt)['status']==('PASS_ENGINEERING_INPUTS' if pilot else 'COMPLETE_UNIT')
  paths.extend(sorted((root/'completed/got').glob('*.json')))
 if missing:
  save(target/'status.json',dict(status='INCOMPLETE_NO_RESULTS_EXPORTED',missing_units=missing,expected_inputs_per_backbone=len(expected),expected_arm_inputs=len(expected)*4));return False
 identities=[]
 for root in roots:
  identity=read(root/'model_cache/model_identity.json')
  backbone=next(b for b in MODELS if ('_'+b+'_') in root.name)
  reference=read(OUT/'generation'/('pilot_'+backbone+'_1729_all_got_0_2')/'model_cache/model_identity.json')
  assert identity==reference,'Runtime identity differs from the validated complete pilot: '+str(root)
  identities.append(dict(path=str(root/'model_cache/model_identity.json'),sha256=sha(root/'model_cache/model_identity.json'),matches_pilot=True))
 save(target/'runtime_identities.json',dict(status='PASS',units=len(identities),identities=identities))
 groups={};costs=[];physical=[];source_files=[];timings={b:[] for b in MODELS};checked=0
 for path in paths:
  row=read(path);rec=row['record'];b=row['backbone'];key=rec['task_id'],rec['input_uid']
  assert key in expected_keys and row['seed']==1729 and row['pair']=='got'
  assert rec==next(r for r in expected if (r['task_id'],r['input_uid'])==key)
  validate_row(row,expected_profile='pilot' if pilot else 'subset_100');checked+=1
  source_files.append(dict(path=str(path),sha256=sha(path)));timings[b].append(row['elapsed_seconds'])
  for name,h in read(OUT/'production_freeze.json')['generation_code'].items():assert row['code_sha256'][Path(name).name]==h
  root=path.parents[2];sim=read(root/'readout_similarity/got'/path.name)
  assert sim['status']=='COMPLETE' and sim['source_sha256']==sha(path)
  source_files.append(dict(path=str(root/'readout_similarity/got'/path.name),sha256=sha(root/'readout_similarity/got'/path.name)))
  for arm in row['arms']:
   for c in arm['pool']:
    fresh=labels(rec,c)
    assert all(c[k]==v for k,v in fresh.items()),(b,key,c['node_id'])
   g=groups.setdefault((b,arm['arm']),{})
   assert key not in g
   g[key]=measures(arm,sim['similarity_full_precision_by_response_id']);costs.append(workload(row,arm))
  unique={r['record_id']:r for a in row['arms'] for r in a['responses']}
  physical.append(dict(backbone=b,seed=1729,task=rec['task_id'],instance_id=rec['input_uid'],physical_response_sequences=len(unique),physical_batched_calls=len({r['physical_batch_id'] for r in unique.values()}),input_tokens=sum(r['input_tokens'] for r in unique.values()),output_tokens=sum(r['output_tokens'] for r in unique.values()),pair_wall_seconds=row['elapsed_seconds']))
 assert checked==len(expected)*2 and len(groups)==4
 assert all(set(g)==expected_keys for g in groups.values())
 st.OUT=RELEASE;st.write=save;boot=st.Bootstrap(expected,'deadline_'+mode)
 inputs=[];tasks=[];summary=[];paired=[];arrays={}
 for (b,arm),g in groups.items():
  ordered=[g[r['task_id'],r['input_uid']] for r in expected]
  for rec,v in zip(expected,ordered):inputs.append(dict(backbone=b,arm=arm,seed=1729,task=rec['task_id'],instance_id=rec['input_uid'],raw_source_id=rec['source_molecule_uid'],**v))
  for metric in ordered[0]:
   vals=np.array([v[metric] for v in ordered],float);arrays[b,arm,metric]=vals
   est,_=boot.estimate(vals);summary.append(dict(backbone=b,arm=arm,seed=1729,metric=metric,n_inputs=len(expected),**est))
   for t,idx in boot.tasks.items():tasks.append(dict(backbone=b,arm=arm,seed=1729,task=t,metric=metric,n_inputs=len(idx),estimate=100*float(vals[idx].mean())))
 for b in MODELS:
  for metric in ['SR@1','SR@5','SR@10','SR@20','SR_sim>0.6@1','SR_sim>0.6@20','validity']:
   paired.append(dict(backbone=b,seed=1729,contrast='GoT minus Sampling with GoT scoring',metric=metric,primary=(metric=='SR@1'),n_inputs=len(expected),**boot.difference(arrays[b,'GoT',metric],arrays[b,'Sampling with GoT scoring',metric])))
 csv_release(target/'per_input_metrics.csv',inputs);csv_release(target/'per_task_metrics.csv',tasks);csv_release(target/'per_seed_summary.csv',summary);csv_release(target/'paired_differences.csv',paired);csv_release(target/'workload.csv',costs);csv_release(target/'physical_workload.csv',physical)
 # Complete candidate and response records, without collapsing empty/invalid/duplicate positions.
 with gzip.open(target/'paired_records.jsonl.gz','wt') as f:
  for path in paths:
   row=read(path);sim=read(path.parents[2]/'readout_similarity/got'/path.name)
   f.write(json.dumps(dict(raw_pair=row,offline_raw_similarity=sim),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
 save(target/'source_files.json',source_files)
 elapsed=time.perf_counter()-started
 save(target/'status.json',dict(status='PASS_COMPLETE_FIXED_SCOPE',mode=mode,inputs_per_backbone=len(expected),paired_inputs=checked,arm_inputs=len(inputs),generation_seeds=[1729],bootstrap=boot.metadata,wall_seconds=elapsed,completed_utc=now().isoformat(),pilot_excluded_from_formal_results=True,all_raw_logs_preserved=True,runtime_identity_checks=len(identities)))
 if pilot:
  remaining=[u for u in read(RELEASE/'units.json') if not (unit_path(u)/'completion.json').exists()]
  reused=[]
  for u in read(RELEASE/'units.json'):
   root=unit_path(u)
   if not (root/'completion.json').exists():continue
   pilot_identity=read(OUT/'generation'/('pilot_'+u['backbone']+'_1729_all_got_0_2')/'model_cache/model_identity.json')
   assert read(root/'model_cache/model_identity.json')==pilot_identity,'Previously generated runtime identity differs from the validated pilot'
   files=sorted((root/'completed/got').glob('*.json'));assert len(files)==u['stop']-u['start']
   for path in files:
    row=read(path);validate_row(row,expected_profile='subset_100')
    assert row['record']['task_id']==u['task'] and u['start']<=row['record']['subset_rank']<u['stop']
    for name,h in read(OUT/'production_freeze.json')['generation_code'].items():assert row['code_sha256'][Path(name).name]==h
    sim=read(root/'readout_similarity/got'/path.name);assert sim['status']=='COMPLETE' and sim['source_sha256']==sha(path)
    reused.append(dict(path=str(path),sha256=sha(path)))
  save(RELEASE/'reused_generation_audit.json',dict(status='PASS',paired_inputs=len(reused),source_files=reused,all_runtime_identities_match_new_pilot=True,no_outcome_based_selection=True))
  per_model={b:dict(n=len(v),mean_seconds=float(np.mean(v)),max_seconds=max(v)) for b,v in timings.items()}
  conservative_gpu_seconds=sum((u['stop']-u['start'])*per_model[u['backbone']]['max_seconds']*cfg['timing_safety_multiplier']+60 for u in remaining)
  gen_seconds=conservative_gpu_seconds/cfg['max_concurrent_gpus']
  reserve=max(cfg['minimum_finish_reserve_seconds'],elapsed*(500/20)*4)
  finish=now()+datetime.timedelta(seconds=gen_seconds+reserve)
  feasible=finish<datetime.datetime.fromisoformat(FREEZE_UTC)
  save(RELEASE/'end_to_end_gate.json',dict(status='PASS_TIMING_AND_PIPELINE' if feasible else 'NO_START_INSUFFICIENT_TIME',checked_utc=now().isoformat(),pilot_paired_inputs=checked,pilot_pipeline_seconds=elapsed,pilot_timing=per_model,remaining_units=len(remaining),reusable_completed_units=40-len(remaining),max_concurrent_gpus=cfg['max_concurrent_gpus'],conservative_remaining_generation_seconds=gen_seconds,reserve_seconds=reserve,latest_budgeted_finish_utc=finish.isoformat(),internal_freeze_utc=FREEZE_UTC,scope_sha256=sha(RELEASE/'scope.json'),formal_started=False,resource_recheck_required=True,scoring_feedback_and_generation_unchanged=True))
 else:
  if now()>datetime.datetime.fromisoformat(FREEZE_UTC):
   save(target/'status.json',dict(status='COMPLETE_AFTER_FREEZE_NOT_FOR_INCLUSION',completed_utc=now().isoformat(),internal_freeze_utc=FREEZE_UTC))
   raise RuntimeError('Completed after the agreed result freeze; do not include automatically.')
 print(mode,'PASS',len(inputs),'arm inputs',round(elapsed,2),'seconds',flush=True)
 return True

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['pilot','generation']);args=p.parse_args()
 try:
  if not summarize(args.mode):raise SystemExit(2)
 except BaseException as e:
  save(RELEASE/(args.mode+'_pipeline_failure.json'),dict(status='INCOMPLETE_OR_FAILED',error=repr(e),utc=now().isoformat()));raise
