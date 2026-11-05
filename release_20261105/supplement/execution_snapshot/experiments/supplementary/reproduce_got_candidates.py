"""Verify saved GoT pairs using the unchanged scorer and success kernel, without GPUs."""
import argparse,ast,csv,gzip,hashlib,importlib.util,json,math,sys,types
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def load(name,path):
 s=importlib.util.spec_from_file_location(name,str(path));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',default=str(Path(__file__).resolve().parent))
 p.add_argument('--pairs',required=True);p.add_argument('--metrics',required=True);p.add_argument('--manifest',required=True);p.add_argument('--expected-count',type=int,required=True);p.add_argument('--output',required=True)
 args=p.parse_args();bundle=Path(args.bundle)
 sources=json.loads((bundle/'evidence/scientific_source_manifest.json').read_text())['copied']
 def source(suffix):
  matches=[r for r in sources if r['path'].endswith('/p6_llama_failure_recovery_20261003_v1/'+suffix)];assert len(matches)==1
  r=matches[0];path=bundle/r['path'];assert sha(path)==r['original_sha256']==r['sha256'];return path
 precision=load('precision',source('code/precision.py'));sys.modules['precision']=precision
 # The production bridge only supplies this metadata constant to the scorer.
 # Read the same binding from its hashed source; do not reimplement score_pool.
 bridge=bundle/'execution_snapshot/experiments/supplementary/bridges.py'
 freeze=json.loads((bundle/'evidence/parent_scientific_freeze.json').read_text())
 expected_bridge=next(v for k,v in freeze['generation_code'].items() if Path(k).name=='bridges.py');assert sha(bridge)==expected_bridge
 values=[n.value.value for n in ast.walk(ast.parse(bridge.read_text())) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='M1' for t in n.targets)]
 assert len(values)==1;shim=types.ModuleType('common');shim.M1=values[0];sys.modules['common']=shim
 scorer=load('verified_original_scorer',source('code/scorer.py'));kernel=load('verified_original_kernel',source('code/metrics_kernel.py'))
 contract=json.loads(source('configs/design_contract.json').read_text());tasks={t['task_id']:t for t in contract['tasks']}
 def csv_rows(path):
  with (gzip.open(path,'rt',newline='') if str(path).endswith('.gz') else open(path,newline='')) as f:return list(csv.DictReader(f))
 fixed=csv_rows(args.manifest);keys={(r['task'],r['instance_id']):r['raw_source_id'] for r in fixed};assert len(keys)==len(fixed)==args.expected_count
 metric_rows=csv_rows(args.metrics);expected={(r['backbone'],r['arm'],r['task'],r['instance_id']):r for r in metric_rows};assert len(expected)==len(metric_rows)==4*args.expected_count
 seen=set();paired=set();positions=0;score_checks=0;success_checks=0;physical=set()
 thresholds=dict(bbbp=.2,drd2=.2,plogp=1.,qed=.1,hia=.1,mutagenicity=.1)
 with gzip.open(args.pairs,'rt') as f:
  for line in f:
   data=json.loads(line);row=data['raw_pair'];rec=row['record'];b=row['backbone'];key=(rec['task_id'],rec['input_uid'])
   assert b in ['mistral','llama'] and row['seed']==1729 and key in keys and rec['source_molecule_uid']==keys[key]
   assert (b,)+key not in paired;paired.add((b,)+key)
   arms=row['arms'];assert [a['arm'] for a in arms]==['GoT','Sampling with GoT scoring']
   assert arms[0]['shared_initial_record_ids']==arms[1]['shared_initial_record_ids'] and len(arms[0]['shared_initial_record_ids'])==12
   assert [c['response'] for c in arms[0]['pool'][:12]]==[c['response'] for c in arms[1]['pool'][:12]]
   similarities=data['offline_raw_similarity']['similarity_full_precision_by_response_id']
   for arm in arms:
    group=(b,arm['arm'])+key;assert group not in seen;seen.add(group);pool=arm['pool'];terminal=arm['candidates']
    assert len(pool)==28 and len(terminal)==20 and [c['original_slot_id'] for c in terminal]==list(range(20))
    byid={c['node_id']:c for c in pool};assert len(byid)==28
    final=scorer.score_pool(pool,byid,rec['source_properties'],tasks[key[0]]['objectives']);assert final==arm['final_score'];score_checks+=1
    assert [c['node_id'] for c in terminal]==final['ranked_node_ids'][:20]
    if arm['arm']=='GoT':
     initial=scorer.score_pool(pool[:12],byid,rec['source_properties'],tasks[key[0]]['objectives']);assert initial==arm['initial_score'];assert initial['ranked_node_ids'][:6]==arm['parent_ids'];score_checks+=1
    else:
     prompt=pool[0]['response']['request']['prompt'];assert all(c['response']['request']['prompt']==prompt and not c['parent_ids'] for c in pool)
    for c in pool:
     assert type(c['rdkit_valid']) is bool;valid=c['rdkit_valid'] and bool(c['raw_smiles'].strip());names=list(rec['source_properties']);props=c['properties']
     finite=kernel.finite_values(props,names);eligible=valid and finite
     if valid and not finite:assert c.get('property_error') or c.get('eval_valid') is False
     success=bool(eligible and kernel.strictly_improves(rec['source_properties'],props,names));delta=bool(success and kernel.meets_delta(rec['source_properties'],props,names,{k:thresholds[k] for k in names}))
     assert c['structural_valid']==valid and c['success']==success and c['threshold_success']==delta
     response=c['response'];rid=response['record_id'];physical.add(rid)
     assert len(response['input_ids'])==response['input_tokens'] and len(response['output_ids'])==response['output_tokens']
     if success:assert rid in similarities and math.isfinite(similarities[rid])
     success_checks+=1
    for c in terminal:
     original=byid[c['node_id']];assert {k:v for k,v in c.items() if k!='original_slot_id'}=={k:v for k,v in original.items() if k!='original_slot_id'}
    vals={}
    for k in [1,5,10,20]:
     vals['SR@'+str(k)]=float(any(c['success'] for c in terminal[:k]))
     vals['SR_sim>0.6@'+str(k)]=float(any(c['success'] and similarities[c['response']['record_id']]>.6 for c in terminal[:k]))
    vals.update(validity=sum(c['structural_valid'] for c in terminal)/20,**{'SR_delta@20':float(any(c['threshold_success'] for c in terminal)),'full28_any_success':float(any(c['success'] for c in pool))})
    for k,v in vals.items():assert abs(v-float(expected[group][k]))<1e-12,(group,k)
    positions+=20
 assert seen==set(expected) and len(paired)==args.expected_count*2 and len(physical)==args.expected_count*88
 report=dict(status='PASS',paired_inputs=len(paired),arm_inputs=len(seen),returned_positions=positions,candidate_success_checks=success_checks,original_scorer_calls=score_checks,physical_response_records=len(physical),pairs_sha256=sha(args.pairs),metrics_sha256=sha(args.metrics),manifest_sha256=sha(args.manifest),scorer_sha256=sha(source('code/scorer.py')),kernel_sha256=sha(source('code/metrics_kernel.py')),scope='Exact saved-cache score and outcome reproduction; no new chemistry, predictor or generation calls.')
 output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
