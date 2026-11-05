"""Lossless flat candidate records and resource summaries for the complete GoT control."""
from deadline_scope import *
import gzip
import argparse

def main():
 require_allocation();parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true');args=parser.parse_args()
 target=RELEASE/('pilot_pipeline' if args.pilot else 'generation');n_inputs=20 if args.pilot else 500
 state=read(target/'status.json');assert state['status']=='PASS_COMPLETE_FIXED_SCOPE'
 assert state['inputs_per_backbone']==n_inputs and state['arm_inputs']==n_inputs*4
 count=0;returned=0;physical_ids=set();arm_keys=set();props=[]
 output=target/'candidate_records.jsonl.gz';tmp=target/'candidate_records.jsonl.gz.tmp'
 with tmp.open('wb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,compresslevel=6,mtime=0) as f:
  for row in lines(target/'paired_records.jsonl.gz'):
   pair=row['raw_pair'];rec=pair['record'];sim=row['offline_raw_similarity']['similarity_full_precision_by_response_id']
   physical_cost=pair['arms'][0]['chemistry_cost']+[c for c in pair['arms'][1]['chemistry_cost'] if not c['shared']]
   assert len(physical_cost)==3
   counters={}
   for c in physical_cost:
    for name,n in c['actual_scalar_computations'].items():counters[name]=counters.get(name,0)+n
   props.append(dict(backbone=pair['backbone'],seed=pair['seed'],task=rec['task_id'],instance_id=rec['input_uid'],candidate_evaluation_attempts=sum(c['candidate_evaluation_attempts'] for c in physical_cost),property_bundle_requests=sum(c['property_bundle_requests'] for c in physical_cost),unique_bundle_requests=sum(c['unique_bundle_requests'] for c in physical_cost),cache_hits=sum(c['cache_hits'] for c in physical_cost),actual_scalar_computations=counters,property_seconds=sum(c['wall_seconds'] for c in physical_cost),measured_evaluator_calls=[c['measured_evaluator_calls'] for c in physical_cost],shared_initial_cost_counted_once=True))
   for arm in pair['arms']:
    arm_key=(pair['backbone'],rec['task_id'],rec['input_uid'],arm['arm']);assert arm_key not in arm_keys;arm_keys.add(arm_key)
    positions={c['candidate_id']:c['original_slot_id'] for c in arm['candidates']};assert sorted(positions.values())==list(range(20))
    assert len(arm['pool'])==28
    for c in arm['pool']:
     response=c['response'];request=response['request'];rid=response['record_id'];physical_ids.add(rid)
     if c['success']:assert rid in sim and isinstance(sim[rid],(float,int)) and math.isfinite(sim[rid])
     data=dict(backbone=pair['backbone'],generation_seed=pair['seed'],task=rec['task_id'],instance_id=rec['input_uid'],raw_source_id=rec['source_molecule_uid'],source_smiles=rec['source_smiles'],source_properties=rec['source_properties'],arm=arm['arm'],pool_position=c['created_order'],returned_position=positions.get(c['candidate_id']),candidate_id=c['candidate_id'],shared_initial_response=c['created_order']<12,full_precision_readout_similarity=sim.get(rid),readout_similarity_status='computed_for_success_constraint' if rid in sim else 'not_required_for_unsuccessful_candidate',candidate=c)
     # The original candidate object retains raw text, prompt, token IDs,
     # random stream, validity, cached values, parent IDs and both scores.
     f.write((json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode())
     count+=1;returned+=int(c['candidate_id'] in positions)
 assert len(arm_keys)==n_inputs*4 and count==n_inputs*112 and returned==n_inputs*80 and len(physical_ids)==n_inputs*88
 os.replace(tmp,output);csv_release(target/'physical_property_workload.csv',props)
 workload_rows=list(csv.DictReader((target/'workload.csv').open()));summary=[]
 fields=['molecular_proposals_attempted','model_response_sequences','physical_batched_model_calls','input_tokens','output_tokens','generation_seconds','property_seconds','candidate_evaluation_attempts','property_bundle_requests','unique_bundle_requests','cache_hits','source_property_gets','similarity_computations','SAS_computations']
 for b in MODELS:
  for arm in ['GoT','Sampling with GoT scoring']:
   rows=[r for r in workload_rows if r['backbone']==b and r['arm']==arm];assert len(rows)==n_inputs
   for field in fields:
    values=[float(r[field]) for r in rows]
    summary.append(dict(backbone=b,arm=arm,metric=field,n_inputs=n_inputs,mean_per_input=sum(values)/n_inputs,total=sum(values),scope='Logical per-arm complete procedure; shared initial generation and evaluation count in each arm.'))
 csv_release(target/'workload_summary.csv',summary)
 save(target/'candidate_export_check.json',dict(status='PASS',mode='engineering_pilot' if args.pilot else 'complete_formal_scope',completed_utc=now().isoformat(),arm_candidate_positions=count,returned_positions=returned,physical_response_records=len(physical_ids),complete_arm_inputs=len(arm_keys),physical_pair_inputs=len(props),candidate_records_sha256=sha(output),paired_records_sha256=sha(target/'paired_records.jsonl.gz'),workload_source_sha256=sha(target/'workload.csv'),exporter_sha256=sha(__file__),original_candidate_objects_preserved=True,missing_labels_imputed=False,physical_property_cost_deduplicates_shared_initial=True))
 print('Candidate export PASS',count,'arm candidate positions;',len(physical_ids),'physical responses',flush=True)

if __name__=='__main__':main()
