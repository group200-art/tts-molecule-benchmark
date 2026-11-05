"""Fixed saved pools. Never recover candidates from aggregate statistics."""
from common import *
from stats import permutation_success,vote_uniform
from collections import Counter

def got():
 for b in MODELS:
  count=[0]
  def rows():
   for row in lines(OUT/'original'/(b+'__got.jsonl.gz')):
    selected=row['candidates'];pool=row['pool'];creation=sorted(selected,key=lambda c:c['created_order']);s20=sum(c['success'] for c in selected);s28=sum(c['success'] for c in pool)
    vals={}
    selectors={'same20_score_order':selected,'same20_creation_order':creation,'full28_score_top20':selected,'full28_creation_first20':pool[:20]}
    for name,cs in selectors.items():
     vals[name]={'SR@'+str(k):float(any(c['success'] for c in cs[:k])) for k in KS};vals[name]['validity']=sum(c['structural_valid'] for c in cs)/20
    for name,n,s,cs in [('same20_uniform_permutation',20,s20,selected),('full28_uniform_permutation',28,s28,pool)]:
     vals[name]={'SR@'+str(k):permutation_success(n,s,k) for k in KS};vals[name]['validity']=sum(c['structural_valid'] for c in cs)/n
    assert all(vals[k]['SR@20']==vals['same20_score_order']['SR@20'] and vals[k]['validity']==vals['same20_score_order']['validity'] for k in ['same20_creation_order','same20_uniform_permutation'])
    o28=int(s28>0);c20=int(s20>0);c1=int(selected[0]['success']);parts=[1-o28,o28-c20,c20-c1];assert sum(parts)==1-c1 and all(p>=0 for p in parts) and sum(p>0 for p in parts)<=1
    vals['first_output_failure_decomposition']=dict(pool_has_no_success=parts[0],selection_omits_success=parts[1],success_returned_but_not_first=parts[2],first_output_failure=1-c1,full28_any_success=o28,full28_validity=sum(c['structural_valid'] for c in pool)/28)
    count[0]+=1
    yield dict(backbone=b,method='got',analysis='got_ordering',record=row['record'],selectors=vals)
  emit(OUT/'offline_inputs'/('got_ordering__'+b+'.jsonl.gz'),rows());write(OUT/'offline_status'/('got_ordering__'+b+'.json'),dict(status='COMPLETE' if count[0]==3905 else 'BLOCKED',inputs=count[0],denominator_required=3905))
  log('got_offline_complete',backbone=b,inputs=count[0])

def vote(backbone,task):
 from chemistry import Chemistry
 from original import normalized
 selector=extracted_function(V/'voting/extract.py',{'count_vote'},dict(Counter=Counter))['count_vote']
 winners={(r['task_id'],r['input_uid'],r['raw_smiles']):r for r in lines(V/'properties/results/winner_properties.jsonl.gz') if r['model_id']=='gellmo_p6_'+backbone and r['task_id']==task}
 root=OUT/'vote_evaluation'/backbone/task;attempt=root/('attempt_'+str(os.environ['SLURM_JOB_ID'])+'_'+str(os.getpid()));attempt.mkdir(parents=True,exist_ok=False)
 chem=Chemistry(task,attempt,'raw');n=0
 try:
  for row in lines(OUT/'original'/('vote_pool_'+backbone+'.jsonl.gz')):
   record=row['record']
   if record['task_id']!=task:continue
   path=root/'inputs'/(record['input_uid']+'.json')
   if path.exists():n+=1;continue
   pool=row['proposals'];byraw={}
   for p in pool:
    raw=p['parsed_candidate']
    if raw in byraw:continue
    key=(task,record['input_uid'],raw)
    if key in winners:
     c=dict(winners[key],candidate_id=p['candidate_id'],original_slot_id=p['global_slot'],raw_text=p['raw_response']);c.update(labels(record,c));byraw[raw]=dict(candidate=c,provenance='original_winner_property_record')
   missing=list(dict.fromkeys(p['parsed_candidate'] for p in pool if p['parsed_candidate'] not in byraw));cost=[]
   for start in range(0,len(missing),100):
    selected=missing[start:start+100];candidates=[]
    for raw in selected:
     p=next(p for p in pool if p['parsed_candidate']==raw)
     candidates.append(dict(candidate_id=p['candidate_id'],input_uid=record['input_uid'],task_id=task,source_smiles=record['source_smiles'],original_slot_id=p['global_slot'],raw_smiles=raw,raw_text=p['raw_response']))
    outcomes,c=chem.evaluate(record,candidates,dict(arm='saved_vote_pool',access='offline',input_uid=record['input_uid'],batch=start//100));cost.append(c)
    for raw,outcome in zip(selected,outcomes):
     candidate=outcome['filled_candidate'];candidate.update(labels(record,candidate));byraw[raw]=dict(candidate=candidate,provenance='new_offline_property_evaluation_of_original_proposal')
   annotated=[]
   for p in pool:
    c=byraw[p['parsed_candidate']]['candidate']
    # Preserve the original two definitions: selector canonical roundtrip versus
    # outcome raw-parse validity. They can legitimately differ.
    annotated.append(dict(original_proposal=p,properties=c['properties'],raw_properties=c['raw_properties'],property_error=c.get('property_error'),structural_valid=c['structural_valid'],success=c['success'],threshold_success=c['threshold_success'],property_provenance=byraw[p['parsed_candidate']]['provenance'],metric_validity_source='original gellmo_inf raw parse',selector_validity_source='original canonical roundtrip identity classifier'))
   group_success=[sum(c['success'] for c in annotated[20*g:20*(g+1)]) for g in range(20)];group_valid=[sum(c['structural_valid'] for c in annotated[20*g:20*(g+1)]) for g in range(20)]
   vote_positions=[];ties=[]
   for g in range(20):
    w,stats=selector(pool[20*g:20*(g+1)]);assert w['candidate_id']==row['winners'][g]['candidate_id'];vote_positions.append(w['global_slot']);ties.append(stats['tie_count']>1)
   choices={'Vote':vote_positions,'First proposal per group':list(range(0,400,20))};vals={}
   for name,positions in choices.items():
    vals[name]={'SR@'+str(k):float(any(annotated[j]['success'] for j in positions[:k])) for k in KS};vals[name]['validity']=sum(annotated[j]['structural_valid'] for j in positions)/20
   vals['Uniform position per group']={'SR@'+str(k):vote_uniform(group_success,k) for k in KS};vals['Uniform position per group']['validity']=sum(group_valid)/400
   vals['Diagnostic group success upper bound']={'SR@'+str(k):float(any(group_success[:k])) for k in KS};vals['Diagnostic group success upper bound']['validity']=None
   valid_classes=Counter(p['canonical_smiles'] for p in pool if p['rdkit_valid_ev']);total_valid=sum(valid_classes.values())
   diag=dict(success_available_groups=sum(s>0 for s in group_success)/20,successful_group_winner_fails=sum(group_success[g]>0 and not annotated[j]['success'] for g,j in enumerate(vote_positions))/20,valid_canonical_unique_classes=len(valid_classes),valid_proposals=total_valid,max_valid_class_share=max(valid_classes.values())/total_valid if total_valid else None,tied_group_fraction=sum(ties)/20,winner_unique_identities=len({(pool[j]['class_type'],pool[j]['class_value']) for j in vote_positions}))
   write(path,dict(backbone=backbone,method='voting',analysis='vote_fixed_pool',record=record,selectors=vals,diagnostics=diag,group_success_counts=group_success,proposals=annotated,workload=cost,original_source_path=row['original_source_path'],original_source_sha256=row['original_source_sha256']))
   n+=1
  write(root/'completion.json',dict(status='COMPLETE' if n==COUNTS[task] else 'BLOCKED',inputs=n,required=COUNTS[task]))
  log('vote_offline_task_complete',backbone=backbone,task=task,inputs=n)
 finally:chem.close()

def main(args):
 require_allocation();assert read(OUT/'reconciliation_report.json')['status']=='PASS'
 analysis=getattr(args,'analysis',None)
 if analysis in (None,'got'):got()
 if analysis=='vote':
  assert args.backbone and args.task;vote(args.backbone,args.task)
 if analysis in (None,'similarity'):
  import subprocess
  subprocess.run([EVAL_PY,'-B',str(CODE/'similarity.py')],check=True)
