"""Real ADMET CPU producer. Launched only inside a pilot allocation."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from contracts import read, write, sha
from property_bridge import finite_or_none
from runtime_common import require_allocation


def main():
    import argparse, os, time, platform
    p=argparse.ArgumentParser();p.add_argument('--requests',required=True);p.add_argument('--output',required=True)
    args=p.parse_args(); require_allocation()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise PermissionError('ADMET must hide GPU before imports')
    import torch, pandas as pd
    from rdkit import Chem, rdBase
    from admet_ai import ADMETModel
    from admet_ai.constants import DEFAULT_MODELS_DIR
    torch.set_num_threads(8)
    if torch.cuda.is_available(): raise RuntimeError('ADMET unexpectedly sees GPU')
    data=read(args.requests); output=Path(args.output); output.mkdir(exist_ok=False)
    requests=data['requests']; names=data['admet_columns']
    rows=[]; valid_strings=[]
    for req in requests:
        raw=req['identity']['raw_smiles']
        mol=Chem.MolFromSmiles(raw) if raw.strip() else None
        valid=mol is not None and mol.GetNumAtoms()>0
        rows.append({'request_id':req['request_id'],'raw_smiles':raw,
                     'valid':bool(valid),'canonical_smiles':Chem.MolToSmiles(mol) if valid else '',
                     'invalid_reason':None if valid else 'empty_or_RDKit_parse_invalid'})
        if valid and raw not in valid_strings: valid_strings.append(raw)
    write(output/'recheck.json',rows)
    detail={'interpreter':sys.executable,'python':platform.python_version(),'torch':torch.__version__,
            'rdkit':rdBase.rdkitVersion,'pandas':pd.__version__,'device':'cpu',
            'model_directory':str(DEFAULT_MODELS_DIR),'num_workers':0,'fingerprint_multiprocessing_min':101,
            'unique_valid_strings':len(valid_strings),'model_load_calls':0,'predict_started':0,'predict_completed':0,
            'load_wall_seconds':None,'predict_wall_seconds':None}
    byraw={}
    if valid_strings:
        if len(valid_strings)>100: raise ValueError('Pilot ADMET batch exceeds visible budget')
        t=time.perf_counter()
        model=ADMETModel(models_dir=DEFAULT_MODELS_DIR,num_workers=0,fingerprint_multiprocessing_min=101)
        detail.update(model_load_calls=1,load_wall_seconds=time.perf_counter()-t,device=str(model.device))
        if str(model.device)!='cpu': raise RuntimeError('ADMET device differs from approved CPU path')
        write(output/'model_loaded.json',detail)
        detail['predict_started']=1;write(output/'predict_started.json',detail)
        t=time.perf_counter(); frame=model.predict(smiles=valid_strings)
        detail.update(predict_completed=1,predict_wall_seconds=time.perf_counter()-t)
        if not isinstance(frame,pd.DataFrame) or list(frame.index)!=valid_strings:
            raise ValueError('ADMET output row identity/order mismatch; no canonical join')
        frame.to_csv(str(output/'admet_raw.csv'),index_label='smiles')
        if any(c not in frame.columns for c in names.values()):
            raise ValueError('Required non-percentile ADMET columns missing')
        for raw in valid_strings:
            byraw[raw]={name:finite_or_none(frame.loc[raw,col]) for name,col in names.items()}
    else:
        pd.DataFrame(columns=list(names.values())).to_csv(str(output/'admet_raw.csv'),index_label='smiles')
    for row in rows:
        row['admet_raw_targets']=byraw.get(row['raw_smiles'],{})
        row['nonfinite_or_missing_fields']=[n for n in names if row['admet_raw_targets'].get(n) is None]
    write(output/'admet_result.json',{'requests_sha256':sha(args.requests),'rows':rows,'identity':detail})


if __name__=='__main__': main()
