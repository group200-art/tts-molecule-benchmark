"""Read-only adapters for the accepted 18 configurations and full saved pools."""
from common import *
from collections import Counter

def graph_paths(backbone):
 if backbone=='mistral':
  # Complete graphs from the original execution and its documented continuations.
  roots=[M,M/'recovery_execution_v1',M/'recovery_execution_v1/bpq_repair_v1']
  for root in roots:
   for p in sorted((root/'workers').glob('*/graphs/G12/*.json')):yield p
 else:
  for unit in read(G/'results/raw_evidence_index.json'):
   if unit['unit']['method_id']=='got':
    for p in sorted((Path(unit['run_directory'])/'graphs').glob('*.json')):yield p

def graph_records(backbone):
 seen={}
 for path in graph_paths(backbone):
  graph=read(path)
  if graph.get('event')!='complete':continue
  r=graph['record'];key=(r['task_id'],r['input_uid'])
  if key in seen:raise ValueError('Duplicate complete GoT graphs: '+str((seen[key],path)))
  seen[key]=str(path)
  assert len(graph['nodes'])==28 and len(graph['top20'])==20 and set(graph['properties'])=={n['node_id'] for n in graph['nodes']}
  assert [n['created_order'] for n in graph['nodes']]==list(range(28))
  pool=[]
  initial_scores={s['node_id']:s for s in graph['scores']['wave0']['scores']}
  final_scores={s['node_id']:s for s in graph['scores']['full']['scores']}
  for n in graph['nodes']:
   p=graph['properties'][n['node_id']]
   c=dict(p,candidate_id=n['node_id'],raw_smiles=n['parsed_smiles'],raw_text=n['raw_response'],created_order=n['created_order'],parent_ids=n['parents'],original_slot_id=n['created_order'],initial_score=initial_scores.get(n['node_id']),final_score=final_scores[n['node_id']],node=n)
   c.update(labels(r,c));pool.append(c)
  byid={c['candidate_id']:c for c in pool}
  selected=[dict(byid[uid],original_slot_id=i) for i,uid in enumerate(graph['top20'])]
  yield r,selected,path,dict(pool=pool,graph=graph)

def vote_rows(backbone):
 yield from lines(V/('voting/grouped_inputs_'+backbone+'.jsonl.gz'))

def vote_full(backbone):
 prev=None;cache=None
 for r in vote_rows(backbone):
  path=Path(r['source_proposal_path'])
  if path!=prev:
   obj=read(path);cache={x['record']['input_uid']:x['classified_proposals'] for x in obj['inputs']} if isinstance(obj,dict) else None
   prev=path
  proposals=cache[r['input_uid']] if cache is not None else obj
  assert len(proposals)==400 and [c['global_slot'] for c in proposals]==list(range(400))
  assert all(c['input_uid']==r['input_uid'] and c['block']==i//20 and c['block_slot']==i%20 for i,c in enumerate(proposals))
  yield r,proposals,path

def standard_records(backbone,method):
 if method=='got':
  yield from graph_records(backbone);return
 if method in ('beam','sampling'):
  old={'beam':'beam20','sampling':'sampling40_prefix20'}[method]
  root=B/'next3_ten_tasks_run01_v1/results/tasks' if backbone=='mistral' else G/'results/by_task'
  for task in TASKS:
   path=root/(old+'__'+task+'.json')
   for r in read(path)['inputs']:yield r,r['candidates'],path,{}
  return
 if method=='self_refine' and backbone=='mistral':
  for task in TASKS:
   path=B/'self_refine_r1_k20_v1/results'/(task+'_K20.json.gz')
   for r in read(path)['rows']:yield r,r['candidates'],path,{}
  return
 if method=='icl' and backbone=='mistral':
  for path in sorted((B/'k20_remaining_three_v1/runs').glob('fewshot_*/result.json.gz')):
   o=read(path);assert o['status']=='complete' and not o['probe']
   for r in o['rows']:yield r,r['candidates'],path,{}
  return
 if method=='voting':
  props={(r['task_id'],r['input_uid'],r['raw_smiles']):r for r in lines(V/'properties/results/winner_properties.jsonl.gz') if r['model_id']=='gellmo_p6_'+backbone}
  path=V/('voting/grouped_inputs_'+backbone+'.jsonl.gz')
  for r in lines(path):
   cs=[dict(props[r['task_id'],r['input_uid'],w['raw_smiles']],candidate_id=w['candidate_id'],original_slot_id=i,raw_text=w['raw_response'],seed=w['seed'],winner=w) for i,w in enumerate(r['winners'])]
   yield r,cs,path,{}
  return
 paths={('llama','self_refine'):R/'llama_self_refine_properties',('mistral','tot'):R/'tot/properties/mistral',('llama','tot'):R/'tot/properties/llama',('mistral','reflexion'):R/'mistral_reflexion_evaluation',('llama','reflexion'):R/'llama_reflexion',('mistral','s1'):V/'next_k20/mistral_s1_properties',('llama','s1'):B/'benchmark_paired_k1_k20_20261014_v1/llama_s1/properties',('llama','icl'):V/'next_k20/llama_icl_properties'}
 path=paths[backbone,method]/'results/terminal_inputs.jsonl.gz'
 for r in lines(path):yield r,r['candidates'],path,{}

def normalized(record,c):
 # Keep physical slots, molecular values and exact read provenance; heavy role traces stay at source paths.
 result={k:c.get(k) for k in ['candidate_id','original_slot_id','raw_smiles','raw_text','raw_response','properties','raw_properties','rdkit_valid','canonical_representations','inf_canonical','eval_valid','property_error','known_molecular_property_failure','missing_reason','environment_validity_disagreement','request_id','rounding_csv','rounding_versions','seed','generation_seed','created_order','parent_ids','initial_score','final_score','parse']}
 result.update(labels(record,c));return result

def inventory():
 require_allocation();expected={(r['task_id'],r['input_uid']):r for r in manifest()};report=[];totals={};errors=[]
 for backbone in MODELS:
  for method in METHODS:
   output=OUT/'original'/('%s__%s.jsonl.gz'%(backbone,method));seen=set();counts=defaultdict(Counter);record_errors=[]
   def rows():
    try:
     for r,cs,path,extra in standard_records(backbone,method):
      key=r['task_id'],r['input_uid'];entry=counts[key[0]]
      if key in seen:raise ValueError('Duplicate input '+str(key))
      seen.add(key);assert key in expected and r['source_properties']==expected[key]['source_properties']
      assert len(cs)==20 and [c['original_slot_id'] for c in cs]==list(range(20))
      entry['inputs']+=1;entry['slots']+=20
      normalized_candidates=[]
      for c in cs:
       entry['raw_smiles']+=int(isinstance(c.get('raw_smiles'),str));entry['raw_response']+=int(isinstance(c.get('raw_text',c.get('raw_response')),str))
       entry['raw_properties']+=int(isinstance(c.get('raw_properties'),dict));entry['precision_trace']+=int(bool(c.get('rounding_csv') or c.get('cache_contract')))
       entry['seed']+=int(c.get('seed',c.get('generation_seed')) is not None)
       normalized_candidates.append(normalized(expected[key],c))
      out=dict(backbone=backbone,method=method,record=expected[key],candidates=normalized_candidates,original_source_path=str(path),original_source_sha256=SOURCES[str(Path(path).resolve())]['sha256'])
      if extra:
       out['pool']=[normalized(expected[key],c) for c in extra['pool']];out['scores']=extra['graph']['scores'];entry['complete_28_nodes']+=1
      yield out
    except Exception as err:record_errors.append(dict(error=repr(err),task_id=locals().get('r',{}).get('task_id'),input_uid=locals().get('r',{}).get('input_uid')))
   emit(output,rows())
   for t in TASKS:
    exp={u for (task,u) in expected if task==t};found={u for (task,u) in seen if task==t}
    report.append(dict(backbone=backbone,method=method,task=t,manifest_inputs=COUNTS[t],found_inputs=len(found),missing_ids=sorted(exp-found),counts=dict(counts[t]),status='complete_returned_candidate_records' if found==exp and not record_errors else 'blocked',errors=record_errors,raw_response_note='NA where missing; candidate SMILES is never used to fabricate raw model text',precision_basis='original round2 comparison fields; raw values not substituted',normalized_path=str(output)))
   totals[backbone+'__'+method]=len(seen);errors.extend(record_errors);log('inventory_configuration',backbone=backbone,method=method,inputs=len(seen),errors=record_errors)
 # Selector reproduction audits full proposals without calling predictors.
 selector=extracted_function(V/'voting/extract.py',{'count_vote'},dict(Counter=Counter))['count_vote']
 vote=[]
 for backbone in MODELS:
  cc=defaultdict(Counter);bad=[]
  def pool_rows():
   try:
    for r,p,path in vote_full(backbone):
     assert sha(path)==r['source_proposal_sha256'] if str(path) not in SOURCES else SOURCES[str(path)]['sha256']==r['source_proposal_sha256']
     for g in range(20):
      w,s=selector(p[20*g:20*(g+1)]);saved=r['winners'][g]
      assert w['candidate_id']==saved['candidate_id'] and s['top_votes']==saved['top_votes'] and s['tie_count']==saved['tie_count']
     cc[r['task_id']]['inputs']+=1;cc[r['task_id']]['proposals']+=400
     yield dict(backbone=backbone,record=expected[r['task_id'],r['input_uid']],proposals=p,winners=r['winners'],original_source_path=str(path),original_source_sha256=r['source_proposal_sha256'])
   except Exception as err:bad.append(repr(err))
  emit(OUT/'original'/('vote_pool_'+backbone+'.jsonl.gz'),pool_rows())
  vote.append(dict(backbone=backbone,counts=dict(cc),selector_original_function=str(V/'voting/extract.py'),errors=bad,property_note='Full pool property availability is audited separately; winners do not imply other proposals have evaluated properties.'))
  log('vote_pool_audit',backbone=backbone,counts=dict(cc),errors=bad)
 report_obj=dict(stage='inventory',configurations=report,full_vote_pools=vote,original_reflexion_feedback_channel='actor receives only reflections; raw feedback goes to reflection prompt',new_reflexion_configuration='both arms explicitly include common raw action/feedback history',unknown_fields='NA; never interpreted as zero/failure',errors=errors,versions=read(OUT/'version_and_hashes.json'))
 write(OUT/'availability_report.json',report_obj);emit(OUT/'sources_manifest.jsonl',SOURCES.values())
 return report_obj

def verify_baseline():
 require_allocation();archived={(r['model'].lower(),r['method_id'],r['task_id'],r['input_uid']):r for r in lines(R/'results/per_input_quality_and_prefix.jsonl.gz')};differences=[];per_task=[];macro=[];n=0
 for b in MODELS:
  for m in METHODS:
   groups=defaultdict(list)
   for row in lines(OUT/'original'/('%s__%s.jsonl.gz'%(b,m))):
    r=row['record'];cs=row['candidates'];old=archived[b,m,r['task_id'],r['input_uid']]
    value={**{'SR_K'+str(k):any(c['success'] for c in cs[:k]) for k in KS},'delta':any(c['threshold_success'] for c in cs),'n_valid_slots':sum(c['structural_valid'] for c in cs)}
    for k,v in value.items():
     if v!=old[k]:differences.append(dict(backbone=b,method=m,task=r['task_id'],input_uid=r['input_uid'],field=k,original=old[k],reproduced=v))
    groups[r['task_id']].append(value);n+=1
   for t in TASKS:
    vals=groups[t];nt=len(vals)
    if nt!=COUNTS[t]:differences.append(dict(backbone=b,method=m,task=t,field='n_inputs',original=COUNTS[t],reproduced=nt))
    per_task.append(dict(backbone=b,method=m,task=t,n_inputs=nt,**{k:100*sum(r[k] for r in vals)/(nt*(20 if k=='n_valid_slots' else 1)) if nt else None for k in ['SR_K1','SR_K5','SR_K10','SR_K20','delta','n_valid_slots']}))
   if sum(len(v) for v in groups.values())==3905:
    macro.append(dict(backbone=b,method=m,**{k:sum(r[k] for r in per_task[-10:])/10 for k in ['SR_K1','SR_K5','SR_K10','SR_K20','delta','n_valid_slots']}))
 csvout(OUT/'baseline_per_task.csv',per_task);csvout(OUT/'baseline_macro.csv',macro)
 report=dict(status='PASS' if not differences and n==70290 else 'RECONCILIATION_REQUIRED',input_records_checked=n,comparisons_per_input=6,differences=differences,original_results_path=str(R/'results/per_input_quality_and_prefix.jsonl.gz'),recomputed_from_candidates=True)
 write(OUT/'reconciliation_report.json',report);log('baseline_reconciliation',status=report['status'],n=n,mismatches=len(differences))
 return report
