"""Self-Refine offline Python3.6 property producer; original source properties are unchanged."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from contracts import ROOT, REPO, read, write, sha, verify, task_contract
from property_bridge import finite_or_none
from runtime_common import require_allocation


def produce(data, admet, output):
    import platform
    import pandas as pd, numpy as np
    from rdkit import Chem, rdBase
    # The unchanged module resolves its own DRD2 and SAS assets relative to __file__.
    sys.path.insert(0,str(REPO/'data/props'))
    import properties
    if sys.version_info[:2]!=(3,6) or pd.__version__!='1.0.5' or np.__version__!='1.18.5':
        raise RuntimeError('Wrong standard property/precision environment')
    byid={r['request_id']:r for r in admet['rows']}
    if len(byid)!=len(admet['rows']) or set(byid)!={r['request_id'] for r in data['requests']}:
        raise ValueError('ADMET request identity mismatch')
    names=[o['property'] for o in task_contract(data['task_id'])['objectives']]
    responses=[]; table=[]
    for req in data['requests']:
        ident=req['identity']; raw=ident['raw_smiles']; ar=byid[req['request_id']]
        if ar['raw_smiles']!=raw: raise ValueError('Raw SMILES changed across environments')
        mol=Chem.MolFromSmiles(raw) if raw.strip() else None
        valid=mol is not None and mol.GetNumAtoms()>0
        vals=None
        if valid and ar['valid']:
            vals=dict(ar['admet_raw_targets'])
            if 'drd2' in names: vals['drd2']=finite_or_none(properties.drd2(raw))
            if 'plogp' in names: vals['plogp']=finite_or_none(properties.penalized_logp(raw))
            if 'qed' in names: vals['qed']=finite_or_none(properties.qed(raw))
            vals['sas']=finite_or_none(properties.sas(raw))
            # Same binary Morgan radius2/2048/chirality=False contract as original evaluator.
            vals['sim']=finite_or_none(properties.similarity(ident['source_smiles'],raw))
        row={'request_id':req['request_id'],'raw_smiles':raw,'source_smiles':ident['source_smiles'],
             'cache_contract':ident['cache_contract'],'status':'hit' if vals is not None else 'missing',
             'rdkit_valid':ar['valid'],'chemistry_basis':'gellmo_inf RDKit recheck; eval validity separately retained',
             'canonical_representations':[
                 {'environment':'gellmo_inf','rdkit_version':admet['identity']['rdkit'],'smiles':ar['canonical_smiles'],'valid':ar['valid']},
                 {'environment':'gellmo_eval_py36','rdkit_version':rdBase.rdkitVersion,'smiles':Chem.MolToSmiles(mol) if valid else '', 'valid':bool(valid)}],
             'environment_validity_disagreement':ar['valid']!=bool(valid),
             'admet_raw_targets':ar['admet_raw_targets'],'values':vals,
             'missing_reason':None if vals is not None else 'invalid_in_inf_or_eval',
             'nonfinite_or_missing_fields':[p for p in names+['sas','sim'] if vals is None or vals.get(p) is None]}
        responses.append(row)
        table.append(dict(request_id=req['request_id'],raw_smiles=raw,**(vals or {})))
    pd.DataFrame(table).to_csv(str(output/'standard_raw.csv'),index=False)
    write(output/'responses_raw.json',responses)
    write(output/'standard_identity.json',{'python':platform.python_version(),'rdkit':rdBase.rdkitVersion,
          'pandas':pd.__version__,'numpy':np.__version__,'properties_file':properties.__file__,
          'properties_sha256':sha(properties.__file__),'device':'cpu','source_target_properties_recomputed':False})


def evaluate_slots(data):
    from property_bridge import backfill
    from metrics_entry import evaluate_registered
    from contracts import kernel
    filled=backfill(data['candidates'],data['bridge']['requests'],data['bridge']['mapping'],
                    data['bridge']['responses'],data['bridge']['cache_contract'])
    records={r['input_uid']:r for r in data['records']}
    out=[]; k=kernel()
    for c in filled:
        record=records[c['input_uid']]
        if c['source_smiles']!=record['source_smiles'] or c['task_id']!=record['task_id']:
            raise ValueError('Source/candidate identity mismatch')
        names=[o['property'] for o in task_contract(record['task_id'])['objectives']]
        if c['rdkit_valid'] and (not k['finite_values'](c['raw_properties'],names) or not k['finite_values'](c['properties'],names)):
            raise RuntimeError('Valid candidate lacks finite required properties; evaluation incomplete')
        native=dict(c,generated_slot_id=c['original_slot_id'],original_slot_id=0,visible_slot_id=0)
        row=evaluate_registered(record['task_id'],record['input_uid'],record['source_properties'],[native],1)
        for suffix in ('dir','delta'):
            row['selected_generated_slot_id_'+suffix]=c['original_slot_id'] if row['SR_'+suffix+'_success'] else None
        out.append(dict(input_uid=c['input_uid'],task_id=c['task_id'],rep_seed=c['rep_seed'],r=c['original_slot_id'],
                        candidate_id=c['candidate_id'],candidate=c,metric_row=row))
    return out


def main():
    import argparse,os,time
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['produce','slots'],required=True)
    p.add_argument('--input',required=True);p.add_argument('--input-sha',required=True)
    p.add_argument('--admet');p.add_argument('--output',required=True);args=p.parse_args()
    require_allocation()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='': raise PermissionError('Offline evaluation requires hidden GPU')
    verify(args.input,args.input_sha);data=read(args.input);output=Path(args.output)
    if args.mode=='produce':
        output.mkdir(exist_ok=False);admet=read(args.admet)
        if admet['requests_sha256']!=args.input_sha:raise ValueError('Cross-environment request SHA mismatch')
        started=time.perf_counter();produce(data,admet,output)
        write(output/'process_measurement.json',dict(property_produce_wall_seconds=time.perf_counter()-started))
    else:
        started=time.perf_counter();rows=evaluate_slots(data)
        write(output,dict(rows=rows,metrics_wall_seconds=time.perf_counter()-started),compact=True)


if __name__=='__main__':main()
