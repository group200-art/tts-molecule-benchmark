"""Run unchanged accepted CPU property producers in a new exclusive scope."""
import argparse,importlib.util,math
from common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('role',choices=['admet','standard','normalize']);p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--admet');a=p.parse_args()
    require_allocation();verify_dependencies();data=read(a.input)
    if a.role=='normalize':
        sys.path.insert(0,str(LLAMA/'src'))
        from property_bridge import normalize_response
        responses=[]
        names=list(data['cache_contract']['calculators'])
        for response in data['responses']:
            r=normalize_response(response)
            versions={x['environment']:x for x in r['canonical_representations']}
            assert versions['gellmo_inf']['rdkit_version']=='2024.03.6'
            assert versions['gellmo_eval_py36']['rdkit_version']=='2020.03.3'
            assert type(r['rdkit_valid']) is bool and r['rdkit_valid']==versions['gellmo_inf']['valid']
            valid=r['rdkit_valid'];values=r['comparison_properties']
            if not valid:assert values is None
            finite=isinstance(values,dict) and all(isinstance(values.get(k),(int,float)) and math.isfinite(values[k]) for k in names)
            known=known_failure(r)
            assert finite or not valid or known,'Unknown valid-candidate property loss '+r['request_id']
            if known:assert values is None
            r['known_molecular_property_failure']=known
            responses.append(r)
        write(a.output,responses);return
    source=RECOVERY/('admet_worker.py' if a.role=='admet' else 'standard_producer.py')
    sys.path.insert(0,str(RECOVERY/'src'))
    spec=importlib.util.spec_from_file_location('accepted_'+a.role,str(source));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.write=write;module.require_allocation=require_allocation
    if a.role=='admet':
        sys.argv=[str(source),'--requests',a.input,'--output',a.output];module.main()
    else:
        target=Path(a.output);target.mkdir(parents=True,exist_ok=False)
        admet=read(a.admet);assert admet['requests_sha256']==sha(a.input)
        module.produce(data,admet,target)

if __name__=='__main__':main()
