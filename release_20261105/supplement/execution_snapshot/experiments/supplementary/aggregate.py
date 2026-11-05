"""Measured results only; no partial denominators and no prewritten conclusions."""
from common import *
from stats import Bootstrap,constrained
import numpy as np
from collections import defaultdict


def ordered(rows,records):
 bykey={(r['record']['task_id'],r['record']['input_uid']):r for r in rows}
 if len(bykey)!=len(rows):raise ValueError('Duplicate input records')
 required={(r['task_id'],r['input_uid']) for r in records}
 if set(bykey)!=required:raise ValueError('Incomplete input set: %d missing, %d extra'%(len(required-set(bykey)),len(set(bykey)-required)))
 return [bykey[r['task_id'],r['input_uid']] for r in records]

def summaries(rows,boot,backbone,method,analysis,selector=None):
 metrics=rows[0]['values'];output=[];tasks=[]
 for metric in metrics:
  if any(r['values'][metric] is None for r in rows):
   if not all(r['values'][metric] is None for r in rows):raise ValueError('Partial metric availability: '+metric)
   output.append(dict(analysis=analysis,backbone=backbone,method=method,selector=selector,metric=metric,estimate=None,ci_low=None,ci_high=None,n_inputs=len(rows)));continue
  scores=np.array([r['values'][metric] for r in rows],dtype=float);est,_=boot.estimate(scores)
  output.append(dict(analysis=analysis,backbone=backbone,method=method,selector=selector,metric=metric,**est,n_inputs=len(rows)))
  for t,idx in boot.tasks.items():tasks.append(dict(analysis=analysis,backbone=backbone,method=method,selector=selector,task=t,metric=metric,estimate=100*float(scores[idx].mean()),n_inputs=len(idx)))
 return output,tasks

def offline_summary():
 records=manifest();boot=Bootstrap(records,'offline_full_3905');summary=[];taskrows=[];differences=[];status=[];input_records=[]
 for b in MODELS:
  for m in METHODS:
   p=OUT/'offline_inputs'/('similarity__%s__%s.jsonl.gz'%(b,m))
   if not p.exists():status.append(dict(analysis='similarity',backbone=b,method=m,status='PENDING'));continue
   try:rows=ordered(list(lines(p)),records)
   except ValueError as e:status.append(dict(analysis='similarity',backbone=b,method=m,status='BLOCKED',error=str(e)));continue
   vals,tasks=summaries(rows,boot,b,m,'similarity');summary.extend(vals);taskrows.extend(tasks)
   a=[r['values']['SR_sim>0.6@20'] for r in rows];z=[r['values']['SR@20'] for r in rows]
   differences.append(dict(analysis='similarity',backbone=b,method=m,contrast='Sim > 0.6 minus unconstrained',metric='SR@20',**boot.difference(a,z),primary=False))
   input_records.extend(dict(analysis='similarity',backbone=b,method=m,task=r['record']['task_id'],instance_id=r['record']['input_uid'],raw_source_id=r['record']['source_molecule_uid'],**r['values']) for r in rows)
   status.append(dict(analysis='similarity',backbone=b,method=m,status='COMPLETE',inputs=len(rows)))
  for analysis,method in [('got_ordering','got'),('vote_fixed_pool','voting')]:
   if method=='got':paths=[OUT/'offline_inputs'/('got_ordering__'+b+'.jsonl.gz')];source=list(lines(paths[0])) if paths[0].exists() else []
   else:
    paths=[OUT/'vote_evaluation'/b/t/'completion.json' for t in TASKS]
    if not all(p.exists() and read(p)['status']=='COMPLETE' for p in paths):status.append(dict(analysis=analysis,backbone=b,status='PENDING'));continue
    source=[]
    for t in TASKS:
     for p in sorted((OUT/'vote_evaluation'/b/t/'inputs').glob('*.json')):
      data=read(p);source.append({k:data[k] for k in ['backbone','method','analysis','record','selectors','diagnostics']})
   try:rows=ordered(source,records)
   except ValueError as e:status.append(dict(analysis=analysis,backbone=b,status='BLOCKED',error=str(e)));continue
   selectors=list(rows[0]['selectors']);score_arrays={}
   for selector in selectors:
    transformed=[dict(r,values=r['selectors'][selector]) for r in rows];s,t=summaries(transformed,boot,b,method,analysis,selector);summary.extend(s);taskrows.extend(t)
    score_arrays[selector]={k:np.array([r['values'][k] for r in transformed],float) for k in transformed[0]['values'] if transformed[0]['values'][k] is not None}
    input_records.extend(dict(analysis=analysis,backbone=b,method=method,selector=selector,task=r['record']['task_id'],instance_id=r['record']['input_uid'],raw_source_id=r['record']['source_molecule_uid'],**r['values']) for r in transformed)
   baseline='Vote' if method=='voting' else 'same20_score_order'
   for selector in selectors:
    if selector==baseline or selector=='first_output_failure_decomposition':continue
    for metric in ['SR@1','SR@5','SR@10','SR@20']:
     differences.append(dict(analysis=analysis,backbone=b,method=method,contrast=selector+' minus '+baseline,metric=metric,**boot.difference(score_arrays[selector][metric],score_arrays[baseline][metric]),primary=False))
   if method=='voting':
    upper=score_arrays['Diagnostic group success upper bound']['SR@20']
    for selector in selectors[:-1]:differences.append(dict(analysis=analysis,backbone=b,method=method,contrast='Diagnostic upper bound minus '+selector,metric='SR@20',**boot.difference(upper,score_arrays[selector]['SR@20']),primary=False))
    csvout(OUT/('vote_pool_diagnostics_'+b+'.csv'),[dict(backbone=b,task=r['record']['task_id'],instance_id=r['record']['input_uid'],**r['diagnostics'],selector_validity_basis='canonical roundtrip identity; used by original Vote',metric_validity_basis='original raw-parse structural validity',metric_valid_proposals=r['selectors']['Uniform position per group']['validity']*400) for r in rows])
    for diagnostic in ['success_available_groups','successful_group_winner_fails','tied_group_fraction']:
     est,_=boot.estimate([r['diagnostics'][diagnostic] for r in rows]);summary.append(dict(analysis='vote_pool_diagnostics',backbone=b,method=method,selector=None,metric=diagnostic,**est,n_inputs=len(rows)))
   status.append(dict(analysis=analysis,backbone=b,status='COMPLETE',inputs=len(rows)))
 csvout(OUT/'offline_per_input_metrics.csv',input_records);csvout(OUT/'offline_per_task_metrics.csv',taskrows);csvout(OUT/'offline_summary.csv',summary);csvout(OUT/'offline_paired_differences.csv',differences)
 for name in ['similarity','got_ordering','vote_fixed_pool']:
  csvout(OUT/({'similarity':'similarity_constraints'}.get(name,name)+'.csv'),[r for r in summary if r['analysis']==name],fields=['analysis','backbone','method','selector','metric','estimate','ci_low','ci_high','n_inputs'])
 write(OUT/'offline_summary_status.json',dict(analyses=status,bootstrap=boot.metadata));return summary,taskrows,differences

def measures(arm,similarities=None):
 cs=arm['candidates'];values={'SR@'+str(k):float(any(c['success'] for c in cs[:k])) for k in KS};values['SR_delta@20']=float(any(c['threshold_success'] for c in cs));values['validity']=sum(c['structural_valid'] for c in cs)/20
 for k in KS:values['SR_sim>0.6@'+str(k)]=float(any(constrained(c['success'],similarities.get(c['response']['record_id']) if similarities is not None else c['similarity_full_precision'],.6) for c in cs[:k]))
 if 'pool' in arm:values['full28_any_success']=float(any(c['success'] for c in arm['pool']))
 if 'trajectories' in arm:
  lanes=arm['trajectories'];first=[l['trials'][0]['success'] for l in lanes]
  values['first_trial_failure_count']=sum(not x for x in first)
  values['recovery_count']=sum(not first[i] and any(t['success'] for t in l['trials'][1:]) for i,l in enumerate(lanes))
  for trial in range(3):values['first_success_trial_'+str(trial+1)]=sum(l['trials'][-1]['success'] and l['trials'][-1]['trial']==trial for l in lanes)/20
  values['final_failure']=sum(not l['trials'][-1]['success'] for l in lanes)/20
 return values

def workload(row,arm):
 responses=arm['responses'];cost=arm['chemistry_cost'];roles=['proposal','thought','action','reflection','judge'];scalar=defaultdict(int);batch=defaultdict(int)
 for c in cost:
  for k,v in c['actual_scalar_computations'].items():scalar[k]+=v
  for k,v in c['actual_batch_calls'].items():batch[k]+=v
 out=dict(backbone=row['backbone'],seed=row['seed'],task=row['record']['task_id'],instance_id=row['record']['input_uid'],arm=arm['arm'],molecular_proposals_attempted=arm['logical_proposals'],model_response_sequences=len(responses),physical_batched_model_calls=len({r['physical_batch_id'] for r in responses}),input_tokens=sum(r['input_tokens'] for r in responses),output_tokens=sum(r['output_tokens'] for r in responses),actual_scalar_computations=dict(scalar),actual_batch_calls=dict(batch),shared_initial_record_ids=arm['shared_initial_record_ids'],arm_wall_seconds=arm.get('logical_arm_wall_seconds'),pair_wall_seconds=row['elapsed_seconds'],hardware_source=row['model_identity_path'])
 out['generation_seconds']=sum({r['physical_batch_id']:r['physical_batch_wall_seconds'] for r in responses}.values())
 out['property_seconds']=sum(c['wall_seconds'] for c in cost)
 out['property_access']='inference'
 out['cost_basis']='logical per-arm complete cost; includes shared initial responses'
 out['measured_evaluator_calls']=[c.get('measured_evaluator_calls') for c in cost]
 for k in ['candidate_evaluation_attempts','property_bundle_requests','unique_bundle_requests','cache_hits','source_property_gets','similarity_computations','SAS_computations']:out[k]=sum(c[k] for c in cost)
 for role in roles:
  selected=[r for r in responses if r['request']['response_role']==role];out[role+'_responses']=len(selected);out[role+'_input_tokens']=sum(r['input_tokens'] for r in selected);out[role+'_output_tokens']=sum(r['output_tokens'] for r in selected)
 return out

def generation_summary():
 expected=subset();groups=defaultdict(list);inputs=[];costs=[];physical=[];record_files=[]
 for unit in sorted((OUT/'generation').glob('subset_100_*')):
  for p in sorted((unit/'completed').glob('*/*.json')):
   row=read(p);record_files.append(p);assert row['profile']=='subset_100'
   sim_record=read(unit/'readout_similarity'/p.parent.name/p.name);assert sim_record['status']=='COMPLETE' and sim_record['source_sha256']==sha(p)
   similarities=sim_record['similarity_full_precision_by_response_id']
   for arm in row['arms']:
    item=dict(backbone=row['backbone'],seed=row['seed'],arm=arm['arm'],pair=row['pair'],record=row['record'],values=measures(arm,similarities));groups[row['backbone'],arm['arm'],row['seed']].append(item)
    inputs.append(dict(backbone=row['backbone'],seed=row['seed'],arm=arm['arm'],task=row['record']['task_id'],instance_id=row['record']['input_uid'],raw_source_id=row['record']['source_molecule_uid'],**item['values']));costs.append(workload(row,arm))
   unique={r['record_id']:r for a in row['arms'] for r in a['responses']};logical=sum(len(a['responses']) for a in row['arms']);saved=logical-len(unique)
   physical_property_cost=row['arms'][0]['chemistry_cost']+[c for c in row['arms'][1]['chemistry_cost'] if not c['shared']]
   physical_scalars=defaultdict(int)
   for c in physical_property_cost:
    for name,n in c['actual_scalar_computations'].items():physical_scalars[name]+=n
   physical.append(dict(backbone=row['backbone'],seed=row['seed'],task=row['record']['task_id'],instance_id=row['record']['input_uid'],pair=row['pair'],physical_responses_completed=len(unique),physical_batched_calls=len({r['physical_batch_id'] for r in unique.values()}),physical_input_tokens=sum(r['input_tokens'] for r in unique.values()),physical_output_tokens=sum(r['output_tokens'] for r in unique.values()),physical_candidate_evaluation_attempts=sum(c['candidate_evaluation_attempts'] for c in physical_property_cost),physical_requested_property_scalar_values=dict(physical_scalars),physical_unique_bundle_requests=sum(c['unique_bundle_requests'] for c in physical_property_cost),shared_property_bundle_requests_saved=sum(c['property_bundle_requests'] for c in row['arms'][1]['chemistry_cost'] if c['shared']),shared_response_sequences_saved=saved,failed_or_uncommitted_attempt_cost='see model_cache/call_events.jsonl; not imputed'))
 def generation_records():
  for p in record_files:
   row=read(p)
   unit=p.parents[2];sim_record=read(unit/'readout_similarity'/p.parent.name/p.name);sim=sim_record['similarity_full_precision_by_response_id']
   for arm in row['arms']:
    molecular=(arm['pool'] if 'pool' in arm else [t['candidate'] for l in arm['trajectories'] for t in l['trials']]);lookup={c['response']['record_id']:c for c in molecular}
    trial_lookup={(l['trajectory'],t['trial']):t for l in arm.get('trajectories',[]) for t in l['trials']}
    terminal_positions={c['response']['record_id']:c['original_slot_id'] for c in arm['candidates']}
    for res in arm['responses']:
     c=lookup.get(res['record_id']);req=res['request']
     if c is not None:
      c=dict(c,**{k:v for k,v in labels(row['record'],c).items() if k not in c});c['similarity_full_precision']=sim.get(res['record_id'])
     trial=trial_lookup.get((req['trajectory_or_node'],req['trial']),{})
     yield dict(experiment_id='recomb-supp-v1',arm=arm['arm'],backbone=row['backbone'],task_id=row['record']['task_id'],instance_id=row['record']['input_uid'],raw_source_id=row['record']['source_molecule_uid'],source_smiles=row['record']['source_smiles'],seed=row['seed'],response_role=req['response_role'],trajectory_or_node=req['trajectory_or_node'],trial=req['trial'],original_position=c.get('original_slot_id') if c else None,parent_ids=c.get('parent_ids') if c else None,raw_response=res['raw_text'],parsed_smiles=c['raw_smiles'] if c else None,candidate=c,prompt_sha256=req['prompt_sha256'],prompt=req['prompt'],input_tokens=res['input_tokens'],output_tokens=res['output_tokens'],response_record_id=res['record_id'],shared_initial_record_id=res['record_id'] if res['record_id'] in arm['shared_initial_record_ids'] else None,source_record=str(p),returned_output_position=terminal_positions.get(res['record_id']),canonical_identity=(c.get('inf_canonical') or next((x.get('smiles') for x in c.get('canonical_representations',[]) if x.get('environment')=='gellmo_inf'),None)) if c else None,parse_status=c.get('parse',{}).get('parse_status',c.get('parse',{}).get('parse_method')) if c else None,structural_valid=c.get('structural_valid') if c else None,finite_complete_properties=c.get('finite_complete_properties') if c else None,success=c.get('success') if c else None,threshold_success=c.get('threshold_success') if c else None,raw_property_values=c.get('raw_properties') if c else None,comparison_property_values=c.get('properties') if c else None,similarity_full_precision=c.get('similarity_full_precision') if c else None,SAS=(c.get('properties') or {}).get('sas') if c else None,initial_score=c.get('initial_score') if c else None,final_score=c.get('final_score') if c else None,evaluator_cache_request_id=c.get('request_id') if c else None,failure_reason=(c.get('property_error') or ('empty_output' if not c['raw_smiles'].strip() else 'invalid_structure' if not c['structural_valid'] else 'known_ineligible_properties' if not c['eligible'] else 'requested_properties_not_all_improved' if not c['success'] else None)) if c else None,feedback=trial.get('feedback'),history_before=trial.get('history_before'),reflection_memory_before=trial.get('memory_before'),parent_feedback=arm.get('parent_feedback'),stop_rule='first_success_or_three_trials' if row['pair']=='reflexion' else 'fixed_28_pool',random_stream_fields=req['random_stream_fields'],derived_seed=req['derived_seed'],output_token_ids=res['output_ids'],physical_batch_id=res['physical_batch_id'])
 emit(OUT/'generation_records.jsonl.gz',generation_records());csvout(OUT/'per_input_metrics.csv',inputs,fields=None if inputs else ['backbone','seed','arm','task','instance_id']);csvout(OUT/'workload.csv',costs,fields=None if costs else ['backbone','seed','arm','task','instance_id']);csvout(OUT/'physical_workload.csv',physical,fields=None if physical else ['backbone','seed','pair','task','instance_id'])
 status=dict(status='PENDING_GENERATION',completed_arm_inputs=len(inputs),required_arm_inputs=23904,configurations=[]);per_task=[];per_seed=[];paired=[];summary=[]
 arms=['GoT','Sampling with GoT scoring','Reflexion with explicit raw feedback','Reflexion without reflection']
 for b in MODELS:
  for a in arms:
   for s in SEEDS:
    rs=groups[b,a,s];status['configurations'].append(dict(backbone=b,arm=a,seed=s,inputs=len(rs),required=996))
 if len(inputs)==23904:
  arrays={key:ordered(rs,expected) for key,rs in groups.items()};extra=[]
  for b in MODELS:
   for task in TASKS:
    idx=np.array([i for i,r in enumerate(expected) if r['task_id']==task]);den=[np.array([r['values']['first_trial_failure_count'] for r in arrays[b,'Reflexion with explicit raw feedback',seed]],float)[idx] for seed in SEEDS]
    if all(d.sum()>0 for d in den):extra.extend((idx,d) for d in den)
  boot=Bootstrap(expected,'generation_subset_996',extra)
  for (b,a,s),rs in arrays.items():
   metrics=[k for k in rs[0]['values'] if not k.endswith('_count')]
   for metric in metrics:
    y=np.array([r['values'][metric] for r in rs]);taskmeans=[]
    for t,idx in boot.tasks.items():
     est=100*float(y[idx].mean());taskmeans.append(est);per_task.append(dict(backbone=b,arm=a,seed=s,task=t,metric=metric,estimate=est,n_inputs=len(idx)))
    per_seed.append(dict(backbone=b,arm=a,seed=s,metric=metric,estimate=sum(taskmeans)/10))
  for b in MODELS:
   for a,control,primary_metric in [('GoT','Sampling with GoT scoring','SR@1'),('Reflexion with explicit raw feedback','Reflexion without reflection','SR@20')]:
    metrics=[k for k in arrays[b,a,SEEDS[0]][0]['values'] if not k.endswith('_count')]
    for metric in metrics:
     contrasts=[];seed_diffs=[]
     for s in SEEDS:
      y=np.array([r['values'][metric] for r in arrays[b,a,s]]);z=np.array([r['values'][metric] for r in arrays[b,control,s]]);contrasts.append(y-z);est,_=boot.estimate(y-z);seed_diffs.append(est['estimate']);paired.append(dict(backbone=b,arm=a,control=control,metric=metric,seed=s,estimate=est['estimate'],ci_low=None,ci_high=None,primary=metric==primary_metric))
     est,_=boot.estimate(np.mean(contrasts,axis=0));paired.append(dict(backbone=b,arm=a,control=control,metric=metric,seed='mean_of_three',**est,seed_difference_min=min(seed_diffs),seed_difference_max=max(seed_diffs),seed_difference_sample_sd=float(np.std(seed_diffs,ddof=1)),primary=metric==primary_metric))
   for a in arms:
    metrics=[k for k in arrays[b,a,SEEDS[0]][0]['values'] if not k.endswith('_count')]
    for metric in metrics:
     y=np.mean([[r['values'][metric] for r in arrays[b,a,s]] for s in SEEDS],axis=0);est,_=boot.estimate(y);summary.append(dict(backbone=b,arm=a,metric=metric,**est))
  recovery(arrays,expected,boot)
  status['status']='COMPLETE';status['bootstrap']=boot.metadata
 csvout(OUT/'per_task_metrics.csv',per_task,fields=None if per_task else ['backbone','arm','seed','task','metric','estimate']);csvout(OUT/'per_seed_summary.csv',per_seed,fields=None if per_seed else ['backbone','arm','seed','metric','estimate']);csvout(OUT/'paired_differences.csv',paired,fields=None if paired else ['backbone','arm','control','seed','metric','estimate','ci_low','ci_high']);csvout(OUT/'generation_summary.csv',summary,fields=None if summary else ['backbone','arm','metric','estimate','ci_low','ci_high'])
 for filename,arm in [('got_generation_control','GoT'),('reflexion_ablation','Reflexion with explicit raw feedback')]:csvout(OUT/(filename+'.csv'),[r for r in paired if r['arm']==arm],fields=None if paired else ['backbone','arm','control','seed','metric','estimate','ci_low','ci_high'])
 write(OUT/'generation_status.json',status);return status

def recovery(arrays,records,shared_bootstrap):
 output=[];metadata=[]
 for b in MODELS:
  a='Reflexion with explicit raw feedback';c='Reflexion without reflection';taskidx={t:np.array([i for i,r in enumerate(records) if r['task_id']==t]) for t in TASKS}
  F={s:np.array([r['values']['first_trial_failure_count'] for r in arrays[b,a,s]],float) for s in SEEDS}
  for s in SEEDS:assert np.array_equal(F[s],[r['values']['first_trial_failure_count'] for r in arrays[b,c,s]])
  tasks=[t for t,idx in taskidx.items() if all(F[s][idx].sum()>0 for s in SEEDS)];extra=[(taskidx[t],F[s][taskidx[t]]) for t in tasks for s in SEEDS];boot=shared_bootstrap
  reps={};means={};seed_means={}
  for arm in [a,c]:
   perseed=[];bootavg=np.zeros(2000)
   for s in SEEDS:
    H=np.array([r['values']['recovery_count'] for r in arrays[b,arm,s]],float);est=[]
    for t in tasks:
     idx=taskidx[t];den=F[s][idx].sum();value=100*H[idx].sum()/den;est.append(value);output.append(dict(backbone=b,arm=arm,seed=s,task=t,estimate=float(value),first_failure_denominator=int(den)))
     w=boot.taskweights[t];bootavg+=100*(w@H[idx])/(w@F[s][idx])/(len(tasks)*3)
    perseed.append(sum(est)/len(tasks) if tasks else None)
    output.append(dict(backbone=b,arm=arm,seed=s,task='macro_fixed_tasks',estimate=perseed[-1],ci_low=None,ci_high=None))
   seed_means[arm]=perseed
   reps[arm]=bootavg;means[arm]=sum(perseed)/3 if tasks else None
   lo,hi=np.percentile(bootavg,[2.5,97.5]);output.append(dict(backbone=b,arm=arm,seed='mean_of_three',task='macro_fixed_tasks',estimate=means[arm],ci_low=float(lo) if tasks else None,ci_high=float(hi) if tasks else None))
  for j,s in enumerate(SEEDS):
   output.append(dict(backbone=b,arm=a+' minus '+c,seed=s,task='macro_fixed_tasks',estimate=seed_means[a][j]-seed_means[c][j] if tasks else None,ci_low=None,ci_high=None))
  diff=reps[a]-reps[c];lo,hi=np.percentile(diff,[2.5,97.5]);output.append(dict(backbone=b,arm=a+' minus '+c,seed='mean_of_three',task='macro_fixed_tasks',estimate=means[a]-means[c] if tasks else None,ci_low=float(lo) if tasks else None,ci_high=float(hi) if tasks else None))
  metadata.append(dict(backbone=b,included_tasks=tasks,excluded={t:'at least one seed has no first-trial failures' for t in TASKS if t not in tasks},bootstrap=boot.metadata))
 csvout(OUT/'reflexion_recovery.csv',output);write(OUT/'recovery_metadata.json',metadata)

def main(args=None):
 require_allocation();assert read(OUT/'reconciliation_report.json')['status']=='PASS'
 offline_summary();status=generation_summary();write(OUT/'bootstrap_metadata.json',{p.stem:read(p) for p in (OUT/'bootstrap').glob('*.json')});log('aggregate_complete',generation_status=status['status'],arm_inputs=status['completed_arm_inputs'])
if __name__=='__main__':main()
