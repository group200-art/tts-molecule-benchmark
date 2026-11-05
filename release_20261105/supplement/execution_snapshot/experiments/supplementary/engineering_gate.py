"""Freeze production only after every engineering pilot record passes.

No method mean, arm contrast or performance-based choice is computed here.
"""
from common import *
from generation import validate_pair,actor_prompt
from bridges import got_modules,ref_modules
from collections import Counter
import subprocess,difflib

CORE=['common.py','bridges.py','backend.py','generation.py','chemistry.py','property_worker.py','producer_guard.py','property_counts.py','generation_similarity.py','run.py']

def scientific_assets():
 paths=[G/'standard_producer.py',G/'admet_worker.py',G/'src/property_bridge.py',G/'src/property_error_isolation.py',G/'src/contracts.py',G/'configs/design_contract.json',G/'configs/mistral_generation_defaults.json',G/'sources/properties.json',G/'sources/runtime.json',M/'sources/runtime.json',B/'p6_llama_main_ten_tasks_v1/configs/reflexion_prompts.json',B/'p6_llama_main_ten_tasks_v1/configs/properties_reflexion.json',R/'recovery_20261027_v1/property_repair.py']
 paths+=list((REPO/'data/props').glob('*.py'))+list((REPO/'data/props').glob('*.pkl'))+list((REPO/'data/props').glob('*.gz'))
 paths+=list(map(Path,read(OUT/'version_and_hashes.json')['original_code']))
 predictor=read(M/'sources/property_asset_identity_before.json')['files']
 for asset in predictor:
  assert sha(asset['path'])==asset['sha256'], 'Original predictor asset changed: '+asset['path']
  paths.append(Path(asset['path']))
 write(OUT/'predictor_asset_verification.json',dict(status='PASS',assets=predictor))
 return {str(p):sha(p) for p in sorted(set(paths))}

def validate_row(row,expected_profile='pilot'):
 arms=row['arms'];record=row['record'];pair=row['pair'];validate_pair(arms,pair)
 assert row['profile']==expected_profile and row['seed']==1729
 # Actual common responses, not a claim based on sharing an integer seed.
 common=[{r['record_id']:r for r in arm['responses'] if r['record_id'] in arm['shared_initial_record_ids']} for arm in arms]
 assert digest(common[0])==digest(common[1])
 for arm in arms:
  assert len({r['record_id'] for r in arm['responses']})==len(arm['responses'])
  for r in arm['responses']:
   req=r['request'];shared=r['record_id'] in common[0]
   derived,fields=response_seed(1729,row['backbone'],record['task_id'],record['input_uid'],'common_initial' if shared else arm['arm'],req['trajectory_or_node'],req['trial'],req['response_role'])
   assert req['derived_seed']==derived and req['random_stream_fields']==fields
   assert r['record_id']==digest(req) and req['prompt_sha256']==hashlib.sha256(req['prompt'].encode()).hexdigest()
   assert len(r['input_ids'])==r['input_tokens'] and len(r['output_ids'])==r['output_tokens']
   assert 0<r['output_tokens']<=req['max_new_tokens'] and r['random_draw_calls']>=r['output_tokens']
  assert sum(c['candidate_evaluation_attempts'] for c in arm['chemistry_cost'])==arm['logical_proposals']
 if pair=='got':
  scorer,original=got_modules();task=next(t for t in read(G/'configs/design_contract.json')['tasks'] if t['task_id']==record['task_id'])
  for arm in arms:
   pool=arm['pool'];props={c['node_id']:c for c in pool}
   score=scorer.score_pool(pool,props,record['source_properties'],task['objectives']);assert score==arm['final_score']
   assert [c['node_id'] for c in arm['candidates']]==score['ranked_node_ids'][:20]
   assert [c['created_order'] for c in pool]==list(range(28))
  got,sampling=arms;pool=got['pool'];initial=scorer.score_pool(pool[:12],{c['node_id']:c for c in pool[:12]},record['source_properties'],task['objectives']);assert initial==got['initial_score']
  parents=initial['ranked_node_ids'][:6];assert got['parent_ids']==parents
  expected=[[p] for p in parents]+[[parents[j-1] for j in ranks] for ranks in original.PAIRS]+[[parents[j-1] for j in ranks] for ranks in original.GROUPS]
  assert [c['parent_ids'] for c in pool[12:]]==expected
  prompt=sampling['pool'][0]['response']['request']['prompt']
  assert all(c['response']['request']['prompt']==prompt and c['parent_ids']==[] for c in sampling['pool'])
  assert all(c['response']['request']['max_new_tokens']==(200 if i<12 else 220) for a in arms for i,c in enumerate(a['pool']))
 else:
  ref,_,_=ref_modules();templates=read(B/'p6_llama_main_ten_tasks_v1/configs/reflexion_prompts.json');question=ref.task_text(record)
  for arm in arms:
   reflected=arm['arm']=='Reflexion with explicit raw feedback'
   for lane in arm['trajectories']:
    history=[];memory=[];trials=lane['trials']
    for i,t in enumerate(trials):
     assert t['trial']==i and t['history_before']==history and t['memory_before']==memory
     assert t['thought']['request']['prompt']==actor_prompt(templates,question,history,memory)
     assert t['action']['request']['prompt']==actor_prompt(templates,question,history,memory,t['thought'])
     assert t['feedback']['success']==t['success']
     terminal=t['success'] or i==2
     assert bool(t['reflection'])==bool(reflected and not terminal)
     if t['reflection']:memory.append(t['reflection']['raw_text'].strip())
     history.append(dict(raw_action=t['action']['raw_text'],feedback=t['feedback']))
     if terminal:assert i==len(trials)-1
    assert trials[-1]['success'] or len(trials)==3
   assert all(r['request']['max_new_tokens']==250 for r in arm['responses'])
 return dict(backbone=row['backbone'],task=record['task_id'],instance_id=record['input_uid'],pair=pair,status='PASS',elapsed_seconds=row['elapsed_seconds'],logical_responses=[len(a['responses']) for a in arms],physical_response_count=len({r['record_id'] for a in arms for r in a['responses']}))

def main():
 require_allocation();rows=[];expected=subset(profile='pilot');timing={};model_ids={}
 receipts=['reconciliation_report.json','baseline_statistics_reconciliation.json','checks.json','controller_checks.json','stats_checks.json','property_checks.json','guard_checks.json','offline_validation.json','cache_checks.json']
 for name in receipts:assert read(OUT/name)['status']=='PASS',name
 for backbone in MODELS:
  root=OUT/'generation'/('pilot_'+backbone+'_1729_all_both_0_100')
  done=read(root/'completion.json');assert done['status']=='PASS_ENGINEERING_INPUTS' and done['inputs']==20
  identity=read(root/'model_cache/model_identity.json');model_ids[backbone]=identity
  assert identity['generation_config']['temperature']==1 and identity['generation_config']['top_p']==1 and identity['generation_config']['top_k']==0
  for pair in ['got','reflexion']:
   files=list((root/'completed'/pair).glob('*.json'));assert len(files)==20
   byuid={read(p)['record']['input_uid']:read(p) for p in files}
   assert set(byuid)=={r['input_uid'] for r in expected}
   for r in expected:
    row=byuid[r['input_uid']];assert row['record']==r;rows.append(validate_row(row))
   times=[row['elapsed_seconds'] for row in byuid.values()];timing[backbone+'_'+pair]=dict(pilot_inputs=len(times),mean_pair_seconds=sum(times)/len(times),min_pair_seconds=min(times),max_pair_seconds=max(times),extrapolated_996_three_seed_pair_hours=sum(times)/len(times)*996*3/3600,scope='Engineering throughput only; not a method efficiency comparison or guaranteed completion time')
  subprocess.run([EVAL_PY,'-B',str(CODE/'generation_similarity.py'),'--unit',str(root)],check=True)
 # Original numeric implementation and asset hashes must remain unchanged.
 for path,expected_hash in read(OUT/'version_and_hashes.json')['original_code'].items():assert sha(path)==expected_hash
 assets=read(OUT/'checkpoint_tokenizer_hashes.json')
 for item in assets:
  st=Path(item['path']).stat();assert (st.st_size,st.st_mtime_ns)==(item['bytes'],item['mtime_ns'])
 report=dict(status='PASS',paired_inputs=len(rows),required_paired_inputs=80,arm_inputs=160,checks=rows,resources=timing,model_identities=model_ids,pilot_excluded_from_production=True,no_outcome_based_tuning=True)
 write(OUT/'engineering_gate.json',report)
 templates=read(B/'p6_llama_main_ten_tasks_v1/configs/reflexion_prompts.json')
 # Render an actual later actor context, retaining the same thought and memory.
 # Only the raw-feedback prefix differs from the original actor template.
 example=None
 for path in sorted((OUT/'generation/pilot_mistral_1729_all_both_0_100/completed/reflexion').glob('*.json')):
  row=read(path)
  for lane in row['arms'][0]['trajectories']:
   if len(lane['trials'])>1:example=(path,row,lane['trials'][1]);break
  if example:break
 assert example is not None
 path,row,trial=example;ref,_,_=ref_modules();memory=trial['memory_before'];reflections=(templates['experience_header']+'Reflections:\n- '+'\n- '.join(x.strip() for x in memory)) if memory else ''
 original_prompt=templates['actor'].format(reflections=reflections,question=ref.task_text(row['record']),scratchpad='\nThought: '+trial['thought']['raw_text'].strip()+'\nAction:')
 actual=trial['action']['request']['prompt']
 diff=''.join(difflib.unified_diff(original_prompt.splitlines(True),actual.splitlines(True),fromfile='original actor template rendered on same context',tofile='actual matched full-arm actor prompt'))
 safe(OUT/'reflexion_prompt_change.diff').write_text(diff)
 write(OUT/'reflexion_prompt_diff_provenance.json',dict(source_path=str(path),source_sha256=sha(path),trajectory=trial['trajectory'],trial=trial['trial'],actual_prompt_sha256=hashlib.sha256(actual.encode()).hexdigest(),original_template_sha256=sha(B/'p6_llama_main_ten_tasks_v1/configs/reflexion_prompts.json'),original_rendering_is_not_an_additional_generation_run=True))
 freeze=dict(status='APPROVED_ENGINEERING_GATE',meaning='Implementation gate under the user-authorized fixed protocol; not a substitute for user approval of substantive plan changes',profile='subset_100',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),generation_code={str(CODE/name):sha(CODE/name) for name in CORE},scientific_assets=scientific_assets(),model_hashes=assets,config_sha256=sha(OUT/'resolved_config.yaml'),subset_sha256=sha(OUT/'subset_manifest.jsonl'),engineering_gate_sha256=sha(OUT/'engineering_gate.json'),review_condition='Any substantive change to controls, definitions, scope or statistics requires user review before submission.',new_unknown_property_failure_policy=False)
 write(OUT/'production_freeze.json',freeze);log('engineering_gate_passed',paired_inputs=80,production_submitted=False)
if __name__=='__main__':main()
