"""Four matched supplementary arms; no reads of old success labels."""
from common import *
import copy, difflib, types
from bridges import got_modules,ref_modules
from chemistry import Chemistry
from backend import Model

VERSION='supp-generation-v1'

def request(record,backbone,seed,arm,node,trial,role,prompt,cap):
 derived,fields=response_seed(seed,backbone,record['task_id'],record['input_uid'],arm,node,trial,role)
 return dict(version=VERSION,base_run_seed=seed,backbone=backbone,task_id=record['task_id'],instance_id=record['input_uid'],arm=arm,trajectory_or_node=node,trial=trial,response_role=role,prompt=prompt,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),max_new_tokens=cap,derived_seed=derived,random_stream_fields=fields)

def candidate(record,response,slot,parser):
 parsed=parser(response['raw_text']);raw=parsed['parsed_smiles']
 return dict(candidate_id='supp-candidate-'+response['record_id'],input_uid=record['input_uid'],task_id=record['task_id'],source_smiles=record['source_smiles'],original_slot_id=slot,raw_smiles=raw,raw_text=response['raw_text'],parse=parsed,response=response)

def got_pair(record,backbone,seed,model,chem):
 initial_started=time.perf_counter()
 scorer,original=got_modules();task=next(t for t in read(G/'configs/design_contract.json')['tasks'] if t['task_id']==record['task_id'])
 ctx=types.SimpleNamespace(record=record,G=12,task=task,byid={},props={},score_maps={},feedback={},pending=None)
 prompter=original.MolecularPrompter(ctx);initial_prompt=prompter.render('Generate',[],'g00')
 initial_responses=model.call([request(record,backbone,seed,'common_initial','g%02d'%i,0,'proposal',initial_prompt,200) for i in range(12)])
 initial=[]
 for i,res in enumerate(initial_responses):
  c=candidate(record,res,i,original.parse_answer);c.update(node_id=c['candidate_id'],local_node_id='g%02d'%i,created_order=i,operation='Generate',parent_ids=[],parents=[],parsed_smiles=c['raw_smiles']);initial.append(c)
 evaluated,initial_cost=chem.evaluate(record,initial,dict(arm='common_initial',family='got',stage='initial',input_uid=record['input_uid']))
 props={o['filled_candidate']['candidate_id']:o['filled_candidate'] for o in evaluated}
 initial_score=scorer.score_pool(initial,props,record['source_properties'],task['objectives'])
 ctx.byid={c['node_id']:c for c in initial};ctx.props=props;ctx.score_maps={'wave0':{s['node_id']:s for s in initial_score['scores']}}
 parent_ids=initial_score['ranked_node_ids'][:6]
 initial_seconds=time.perf_counter()-initial_started
 results=[]
 for arm in ['GoT','Sampling with GoT scoring']:
  arm_started=time.perf_counter();specs=[]
  if arm=='GoT':
   for i,p in enumerate(parent_ids):specs.append(('Refine','r%02d'%i,[p]))
   for i,ranks in enumerate(original.PAIRS):specs.append(('Merge','m%02d'%i,[parent_ids[j-1] for j in ranks]))
   for i,ranks in enumerate(original.GROUPS):specs.append(('Aggregate','a%02d'%i,[parent_ids[j-1] for j in ranks]))
  else:specs=[('Generate','s%02d'%i,[]) for i in range(16)]
  prompts=[prompter.render(op,[dict(node_id=p) for p in parents],local) for op,local,parents in specs]
  responses=model.call([request(record,backbone,seed,arm,local,0,'proposal',prompt,220) for (op,local,parents),prompt in zip(specs,prompts)])
  later=[]
  for i,(res,(op,local,parents)) in enumerate(zip(responses,specs),12):
   c=candidate(record,res,i,original.parse_answer);c.update(node_id=c['candidate_id'],local_node_id=local,created_order=i,operation=op,parent_ids=parents,parents=parents,parsed_smiles=c['raw_smiles']);later.append(c)
  later_outcome,later_cost=chem.evaluate(record,later,dict(arm=arm,family='got',stage='later',input_uid=record['input_uid']))
  pool=copy.deepcopy(initial)+later;propmap=dict(props,**{o['filled_candidate']['candidate_id']:o['filled_candidate'] for o in later_outcome})
  # Separate complete-pool normalization; no predictor request is made here.
  final=scorer.score_pool(pool,propmap,record['source_properties'],task['objectives']);scores={s['node_id']:s for s in final['scores']}
  enriched=[]
  for c in pool:
   c=dict(propmap[c['candidate_id']],**{k:v for k,v in c.items() if k not in propmap[c['candidate_id']]});c.update(labels(record,c),initial_score=ctx.score_maps['wave0'].get(c['node_id']) if arm=='GoT' else None,final_score=scores[c['node_id']],shared_initial_record_id=c['response']['record_id'] if c['created_order']<12 else None,similarity_full_precision=(c.get('raw_properties') or {}).get('sim'))
   enriched.append(c)
  byid={c['node_id']:c for c in enriched};terminal=[dict(byid[n],original_slot_id=i) for i,n in enumerate(final['ranked_node_ids'][:20])]
  results.append(dict(arm=arm,pool=enriched,candidates=terminal,initial_score=initial_score if arm=='GoT' else None,final_score=final,parent_ids=parent_ids if arm=='GoT' else [],parent_feedback=copy.deepcopy(ctx.feedback) if arm=='GoT' else {},chemistry_cost=[dict(initial_cost,shared=True),dict(later_cost,shared=False)],responses=initial_responses+responses,shared_initial_record_ids=[x['record_id'] for x in initial_responses],logical_proposals=28,logical_arm_wall_seconds=initial_seconds+time.perf_counter()-arm_started,shared_initial_wall_seconds=initial_seconds))
 assert [digest(c['response']) for c in results[0]['pool'][:12]]==[digest(c['response']) for c in results[1]['pool'][:12]]
 return results


def actor_prompt(templates,question,history,memory,thought=None):
 # Raw action and restricted binary/empty/invalid feedback are shown in BOTH arms.
 raw=('Previous actions and evaluator feedback:\n'+'\n\n'.join('Trial %d action:\n%s\nObservation: %s'%(i+1,h['raw_action'],h['feedback']['observation']) for i,h in enumerate(history))+'\n\n') if history else ''
 reflections=(templates['experience_header']+'Reflections:\n- '+'\n- '.join(x.strip() for x in memory)) if memory else ''
 scratchpad='\nThought:'+((' '+thought['raw_text'].strip()+'\nAction:') if thought is not None else '')
 return templates['actor'].format(reflections=raw+reflections,question=question,scratchpad=scratchpad)

def reflexion_pair(record,backbone,seed,model,chem):
 original,bridge,svc=ref_modules();templates=read(B/'p6_llama_main_ten_tasks_v1/configs/reflexion_prompts.json');question=original.task_text(record)
 def round_calls(arm,trial,lanes):
  thoughts=model.call([request(record,backbone,seed,arm,l['trajectory'],trial,'thought',actor_prompt(templates,question,l['history'],l['memory']),250) for l in lanes])
  actions=model.call([request(record,backbone,seed,arm,l['trajectory'],trial,'action',actor_prompt(templates,question,l['history'],l['memory'],thought),250) for l,thought in zip(lanes,thoughts)])
  candidates=[candidate(record,a,l['trajectory'],original.parse_action) for l,a in zip(lanes,actions)]
  outcomes,cost=chem.evaluate(record,candidates,dict(arm=arm,family='reflexion',trial=trial,input_uid=record['input_uid']))
  trials=[]
  for l,th,act,c,outcome in zip(lanes,thoughts,actions,candidates,outcomes):
   assert outcome['feedback']['success']==outcome['success']
   trial_record=dict(trial=trial,trajectory=l['trajectory'],thought=th,action=act,candidate=outcome['filled_candidate'],feedback=outcome['feedback'],reflection=None,history_before=copy.deepcopy(l['history']),memory_before=list(l['memory']),**{k:outcome[k] for k in ['structural_valid','eligible','success','threshold_success','finite_complete_properties']})
   trials.append(trial_record)
  return trials,cost
 initial_started=time.perf_counter()
 initial_lanes=[dict(trajectory=i,history=[],memory=[],trials=[]) for i in range(20)]
 common,initial_cost=round_calls('common_initial',0,initial_lanes)
 initial_seconds=time.perf_counter()-initial_started
 shared_ids=[response['record_id'] for trial in common for response in (trial['thought'],trial['action'])]
 results=[]
 for arm,has_reflection in [('Reflexion with explicit raw feedback',True),('Reflexion without reflection',False)]:
  arm_started=time.perf_counter()
  lanes=copy.deepcopy(initial_lanes);costs=[dict(initial_cost,shared=True)];all_responses=[]
  def finish_trials(active,trials):
   pending=[]
   for lane,tr in zip(active,trials):
    lane['trials'].append(tr);lane['history'].append(dict(raw_action=tr['action']['raw_text'],feedback=tr['feedback']))
    all_responses.extend([tr['thought'],tr['action']]);lane['done']=tr['success'] or tr['trial']==2
    if has_reflection and not lane['done']:pending.append((lane,tr))
   if pending:
    requests=[]
    for lane,tr in pending:
     scratch='\nThought: '+tr['thought']['raw_text'].strip()+'\nAction: '+tr['action']['raw_text'].strip()+'\nObservation: '+tr['feedback']['observation']
     prompt=templates['reflection'].format(question=question,scratchpad=scratch)
     requests.append(request(record,backbone,seed,arm,lane['trajectory'],tr['trial'],'reflection',prompt,250))
    reflections=model.call(requests)
    for (lane,tr),res in zip(pending,reflections):
     tr['reflection']=res;lane['memory'].append(res['raw_text'].strip());all_responses.append(res)
  finish_trials(lanes,copy.deepcopy(common))
  for trial in [1,2]:
   active=[l for l in lanes if not l['done']]
   if not active:break
   trials,cost=round_calls(arm,trial,active);costs.append(dict(cost,shared=False));finish_trials(active,trials)
  terminal=[]
  for lane in lanes:
   tr=lane['trials'][-1];c=dict(tr['candidate'],original_slot_id=lane['trajectory'],trajectory=lane['trajectory'],trial=tr['trial'],shared_initial_record_id=common[lane['trajectory']]['action']['record_id'],similarity_full_precision=(tr['candidate'].get('raw_properties') or {}).get('sim'));c.update(labels(record,c));terminal.append(c)
   assert 1<=len(lane['trials'])<=3 and all(not x['success'] for x in lane['trials'][:-1])
   if not has_reflection:assert all(t['reflection'] is None for t in lane['trials'])
   for t in lane['trials']:
    if t['trial']==2 or t['success']:assert t['reflection'] is None
    if t['trial']>0:
     for h in t['history_before']:assert h['raw_action'] in t['action']['request']['prompt'] and h['feedback']['observation'] in t['action']['request']['prompt']
  results.append(dict(arm=arm,candidates=terminal,trajectories=lanes,chemistry_cost=costs,responses=all_responses,shared_initial_record_ids=shared_ids,logical_proposals=sum(len(l['trials']) for l in lanes),explicit_raw_feedback=True,logical_arm_wall_seconds=initial_seconds+time.perf_counter()-arm_started,shared_initial_wall_seconds=initial_seconds))
 for a,b in zip(results[0]['trajectories'],results[1]['trajectories']):
  for field in ['thought','action','candidate','feedback','success']:assert digest(a['trials'][0][field])==digest(b['trials'][0][field])
 return results

def validate_pair(results,pair):
 assert len(results)==2 and all(len(r['candidates'])==20 for r in results)
 assert results[0]['shared_initial_record_ids']==results[1]['shared_initial_record_ids']
 for r in results:
  assert [c['original_slot_id'] for c in r['candidates']]==list(range(20))
  for c in r['candidates']:
   assert not c['threshold_success'] or c['success']
   if c['success']:assert isinstance(c['similarity_full_precision'],(int,float)) and math.isfinite(c['similarity_full_precision'])
  if pair=='got':assert len(r['pool'])==28
 return True

def main(args):
 require_allocation();assert args.backbone
 assert args.profile in ['pilot','subset_100'], 'Only the frozen 996-input profile is authorized'
 for asset in read(OUT/'checkpoint_tokenizer_hashes.json'):
  stat=Path(asset['path']).stat()
  assert stat.st_size==asset['bytes'] and stat.st_mtime_ns==asset['mtime_ns'], 'Model/tokenizer asset changed; rehash and review before generation'
 assert read(OUT/'reconciliation_report.json')['status']=='PASS'
 for receipt in ['checks.json','controller_checks.json','property_checks.json']:assert read(OUT/receipt)['status']=='PASS'
 for path,expected in read(OUT/'version_and_hashes.json')['original_code'].items():assert sha(path)==expected, 'Original scientific source changed: '+path
 if args.profile!='pilot':
  gate=read(OUT/'production_freeze.json');assert gate['status']=='APPROVED_ENGINEERING_GATE'
  for path,expected in gate['generation_code'].items():assert sha(path)==expected, 'Frozen supplementary source changed'
  for path,expected in gate['scientific_assets'].items():assert sha(path)==expected, 'Frozen original evaluator/prompt asset changed'
  assert gate['config_sha256']==sha(OUT/'resolved_config.yaml') and gate['subset_sha256']==sha(OUT/'subset_manifest.jsonl')
 profile=args.profile;rows=subset(profile=profile);rows=[r for r in rows if args.task is None or r['task_id']==args.task]
 rows=[r for r in rows if args.start<=r['subset_rank']<args.stop]
 seed=args.seed or 1729;pairs=[args.pair] if args.pair else ['got','reflexion']
 unit='_'.join(map(str,[profile,args.backbone,seed,args.task or 'all',args.pair or 'both',args.start,args.stop]))
 root=OUT/'generation'/unit;root.mkdir(parents=True,exist_ok=True)
 # Per-allocation identity avoids overwriting logs of a failed/restarted process.
 attempt=root/('attempt_'+str(os.environ['SLURM_JOB_ID'])+'_'+str(os.getpid()));attempt.mkdir(exist_ok=False)
 for key in ['HF_HOME','HF_HUB_CACHE','TRANSFORMERS_CACHE','XDG_CACHE_HOME','TORCH_HOME','MPLCONFIGDIR','TRITON_CACHE_DIR','CUDA_CACHE_PATH','NUMBA_CACHE_DIR','TMPDIR']:
  path=attempt/'environment'/key;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
 loaded_code_sha={p.name:sha(p) for p in CODE.glob('*.py')}
 write(attempt/'code_identity.json',dict(loaded_code_sha256=loaded_code_sha,production_freeze_sha256=sha(OUT/'production_freeze.json') if profile!='pilot' else None))
 model=None;chem=None
 try:
  model=Model(args.backbone,root/'model_cache')
  for task in TASKS:
   relevant=[r for r in rows if r['task_id']==task]
   if not relevant:continue
   for pair in pairs:
    chem=Chemistry(task,attempt/(pair+'_'+task),'got' if pair=='got' else 'raw')
    for record in relevant:
     if profile!='pilot' and (OUT/'production_hold.json').exists():raise RuntimeError('Another worker reported an error; stopping between inputs for inspection')
     dest=root/'completed'/pair/(record['input_uid']+'.json')
     if dest.exists():
      done=read(dest);assert done['record']==record and done['seed']==seed;validate_pair(done['arms'],pair);continue
     started=time.perf_counter();results=(got_pair if pair=='got' else reflexion_pair)(record,args.backbone,seed,model,chem);validate_pair(results,pair)
     payload=dict(schema=VERSION,experiment_id='recomb-supp-v1',profile=profile,backbone=args.backbone,seed=seed,pair=pair,record=record,arms=results,model_identity_path=str(root/'model_cache/model_identity.json'),elapsed_seconds=time.perf_counter()-started,code_sha256=loaded_code_sha)
     write(dest,payload);log('generated_pair',profile=profile,backbone=args.backbone,seed=seed,task=task,input_idx=record['input_idx'],pair=pair,seconds=payload['elapsed_seconds'])
    chem.close();chem=None
  import subprocess
  subprocess.run([EVAL_PY,'-B',str(CODE/'generation_similarity.py'),'--unit',str(root)],check=True)
  write(root/'completion.json',dict(status='PASS_ENGINEERING_INPUTS' if profile=='pilot' else 'COMPLETE_UNIT',profile=profile,inputs=len(rows),pairs=pairs,seed=seed,backbone=args.backbone,unit=unit))
 finally:
  if chem is not None:chem.close()
