"""Original property APIs and precision chain; failures propagate (Python 3.6)."""
import argparse
from pathlib import Path
import platform
import sys
import time
import traceback
from contracts import ROOT,REPO,read,write,sha,verify,require_allocation,task_contract
from property_bridge import finite_or_none,backfill
from registered_metrics import evaluate_registered,aggregate_registered


def produce(data,chemistry,out):
    import pandas as pd,numpy as np
    from rdkit import Chem,rdBase
    if sys.version_info[:2]!=(3,6) or pd.__version__!='1.0.5' or np.__version__!='1.18.5':raise RuntimeError('Wrong original precision environment')
    sys.path.insert(0,str(REPO/'data/props'));import properties
    names=[o['property'] for o in task_contract(data['task_id'])['objectives']]
    rows=chemistry['rows']
    if [r['request_id'] for r in rows]!=[r['request_id'] for r in data['requests']]:raise ValueError('Cross-environment request ID/order mismatch')
    responses=[];table=[];costs=[]
    for req,ar in zip(data['requests'],rows):
        ident=req['identity'];raw=ident['raw_smiles']
        if ar['raw_smiles']!=raw:raise ValueError('Cross-environment raw mismatch')
        mol=Chem.MolFromSmiles(raw) if raw.strip() else None
        valid=mol is not None and mol.GetNumAtoms()>0;vals=None
        if valid and ar['valid']:
            vals=dict(ar['admet_raw_targets'])
            functions=[(n,getattr(properties,{'plogp':'penalized_logp'}.get(n,n)),(raw,)) for n in names if n in ('drd2','plogp','qed')]
            functions.extend([('sas',properties.sas,(raw,)),('sim',properties.similarity,(ident['source_smiles'],raw))])
            for name,fn,args in functions:
                started={'request_id':req['request_id'],'tool':name,'started':True,'completed':False}
                from contracts import append_event
                append_event(out/'tools.jsonl',dict(started,event='started'));t=time.perf_counter()
                try:
                    # Exceptions are execution failures; only normal nonfinite returns become null.
                    value=fn(*args);vals[name]=finite_or_none(value);started['completed']=True
                    started['raw_return_repr']=repr(value);started['normalized_finite_value']=vals[name]
                except BaseException as exc:started['exception']=repr(exc);raise
                finally:
                    started['wall_seconds']=time.perf_counter()-t
                    append_event(out/'tools.jsonl',dict(started,event='result'));costs.append(started)
        response={'request_id':req['request_id'],'raw_smiles':raw,'source_smiles':ident['source_smiles'],
                  'cache_contract':ident['cache_contract'],'status':'hit' if vals is not None else 'missing','rdkit_valid':ar['valid'],
                  'chemistry_basis':'gellmo_inf RDKit recheck; eval validity separately retained',
                  'canonical_representations':[{'environment':'gellmo_inf','rdkit_version':chemistry['identity']['rdkit'],'smiles':ar['canonical_smiles'],'valid':ar['valid']},
                                               {'environment':'gellmo_eval_py36','rdkit_version':rdBase.rdkitVersion,'smiles':Chem.MolToSmiles(mol) if valid else '', 'valid':bool(valid)}],
                  'environment_validity_disagreement':ar['valid']!=bool(valid),'admet_raw_targets':ar['admet_raw_targets'],'values':vals,
                  'missing_reason':None if vals is not None else 'invalid_in_inf_or_eval',
                  'nonfinite_or_missing_fields':[p for p in names+['sas','sim'] if vals is None or vals.get(p) is None]}
        responses.append(response);table.append(dict(request_id=req['request_id'],raw_smiles=raw,**(vals or {})))
    pd.DataFrame(table).to_csv(str(out/'standard_raw.csv'),index=False)
    write(out/'responses_raw.json',responses)
    write(out/'identity.json',{'python':platform.python_version(),'rdkit':rdBase.rdkitVersion,'pandas':pd.__version__,'numpy':np.__version__,
          'properties_file':properties.__file__,'properties_sha256':sha(properties.__file__),'device':'cpu','source_target_properties_recomputed':False,
          'nested_cost_disclosure':'penalized_logp internally calls SAS; direct timers include nested work and must not be summed with nested time'})


def metrics(data,out):
    b=data['bridge'];filled=backfill(b['candidates'],b['requests'],b['mapping'],b['responses'],b['cache_contract'])
    byid={c['candidate_id']:c for c in filled};byuid={r['input_uid']:r for r in data['records']};groups={str(n):[] for n in (1,20,100,400)}
    for selection in data['selections']:
        record=byuid[selection['input_uid']];n=selection['N_used'];c=dict(byid[selection['candidate_id']])
        c['N_used']=n
        if c['original_slot_id']!=0 or c['proposal_global_slot']>=n:raise ValueError('Native projection mismatch')
        row=evaluate_registered(record['task_id'],record['input_uid'],record['source_properties'],[c],1)
        row.update(N_used=n,proposal_global_slot=c['proposal_global_slot'],vote_candidate_id=c['candidate_id'])
        groups[str(n)].append(row)
    write(out,{'task_id':data['task_id'],'by_budget':{n:{'inputs':rows,'metrics':aggregate_registered(rows)} for n,rows in groups.items()},
               'visible_k':1,'precision':'raw_then_pandas_numpy_round2_csv_v1','basis':'Vote winner only; native evaluation slot0 preserves candidate_id/global slot'})


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['produce','metrics'],required=True);p.add_argument('--input',required=True)
    p.add_argument('--input-sha',required=True);p.add_argument('--chemistry');p.add_argument('--output',required=True);args=p.parse_args()
    require_allocation();verify(args.input,args.input_sha);data=read(args.input);out=Path(args.output)
    if args.mode=='produce':
        out.mkdir(exist_ok=False)
        try:
            chemistry=read(args.chemistry)
            if chemistry['requests_sha256']!=args.input_sha:raise ValueError('Request SHA mismatch')
            produce(data,chemistry,out)
        except BaseException as exc:
            write(out/'exception.json',{'exception':repr(exc),'traceback':traceback.format_exc()});raise
    else:metrics(data,out)


if __name__=='__main__':main()
