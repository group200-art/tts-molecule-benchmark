"""CPU-only existing property/rounding kernels and native-output readouts."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from contracts import ROOT,read,write,verify,unit_for_id,manifest,pilot_config,kernel,task_contract
from runtime_common import require_allocation


def enforce_native_property_gate(method,record,candidate):
    if method=='self_refine':
        # Exact original P5 evaluate_slots finite-property guard, before native metrics.
        names=[o['property'] for o in task_contract(record['task_id'])['objectives']]
        k=kernel()
        if candidate['rdkit_valid'] and (not k['finite_values'](candidate['raw_properties'],names) or not k['finite_values'](candidate['properties'],names)):
            raise RuntimeError('Valid candidate lacks finite required properties; evaluation incomplete')


def evaluate_native(data):
    from property_bridge import backfill
    from metrics_entry import evaluate_registered,aggregate_registered
    u=unit_for_id(data['unit_id']);records=manifest(u['task_id'],u['unit_id'])
    assert records==data['records'] and data['model_id']=='gellmo_p6_llama'
    K=pilot_config()['methods'][u['method_id']]['visible_k']
    if 'bridge' in data:
        b=data['bridge'];filled=backfill(data['candidates'],b['requests'],b['mapping'],b['responses'],b['cache_contract'])
        groups=[dict(record=r,candidates=[c for c in filled if c['input_uid']==r['input_uid']]) for r in records]
    else:groups=data['filled_groups']
    assert [g['record'] for g in groups]==records
    rows=[]
    for group in groups:
        r=group['record'];cs=group['candidates'];assert len(cs)==K
        for c in cs:enforce_native_property_gate(u['method_id'],r,c)
        native=[dict(c,generated_slot_id=c['original_slot_id'],original_slot_id=i,visible_slot_id=i) for i,c in enumerate(cs)]
        row=evaluate_registered(r['task_id'],r['input_uid'],r['source_properties'],native,K)
        row.update(input_idx=r['input_idx'],model_id='gellmo_p6_llama',method_id=u['method_id'],main_id=pilot_config()['methods'][u['method_id']]['main_id'])
        rows.append(row)
    return dict(unit_id=u['unit_id'],model_id='gellmo_p6_llama',method_id=u['method_id'],task_id=u['task_id'],status='complete',inputs=rows,metrics=aggregate_registered(rows))


def main():
    import argparse,time
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['produce','parents','evaluate_trial','native'],required=True)
    p.add_argument('--input',required=True);p.add_argument('--input-sha',required=True);p.add_argument('--admet');p.add_argument('--output',required=True);a=p.parse_args()
    require_allocation();verify(a.input,a.input_sha);data=read(a.input);output=Path(a.output)
    if a.mode=='produce':
        from standard_producer import produce
        output.mkdir(exist_ok=False);admet=read(a.admet);assert admet['requests_sha256']==a.input_sha
        produce(data,admet,output)
    elif a.mode=='parents':
        from adapters import stage_gate,depth2_plan
        t=time.perf_counter();rows=[]
        for item in data:
            gate=stage_gate(item['record'],item['depth1'],item['bridge'])
            rows.append(dict(item,gate=gate,plans=depth2_plan(item['record'],item['depth1'],item['bridge'],gate)))
        write(output,dict(parents=rows,selection_wall_seconds=time.perf_counter()-t))
    elif a.mode=='evaluate_trial':
        from reflexion_evaluator_source import evaluate_trial
        t=time.perf_counter();value=evaluate_trial(data);value['binary_evaluation_wall_seconds']=time.perf_counter()-t;write(output,value)
    else:write(output,evaluate_native(data))


if __name__=='__main__':main()
