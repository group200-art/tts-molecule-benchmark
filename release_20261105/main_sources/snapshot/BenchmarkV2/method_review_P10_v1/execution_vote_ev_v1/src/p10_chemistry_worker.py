"""Pinned RDKit winner recheck and, when required, one ADMET CPU batch."""
import argparse
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from contracts import read,write,sha,require_allocation
from property_bridge import finite_or_none


def main():
    p=argparse.ArgumentParser();p.add_argument('--requests',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    require_allocation()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU-only predictor')
    from rdkit import Chem,rdBase
    if rdBase.rdkitVersion!='2024.03.6':raise RuntimeError('Fixed chemistry environment changed')
    data=read(args.requests);requests=data['requests'];names=data['admet_columns']
    if not 1<=len(requests)<=100:raise ValueError('At most100 requests per batch')
    if bool(names)!=(data['task_id']!='DPQ'):raise ValueError('DPQ ADMET boundary')
    out=Path(args.output);out.mkdir(exist_ok=False)
    detail={'interpreter':sys.executable,'python':platform.python_version(),'rdkit':rdBase.rdkitVersion,'device':'cpu',
            'model_load_started':0,'model_load_completed':0,'predict_started':0,'predict_completed':0,
            'load_wall_seconds':None,'predict_wall_seconds':None,'admet_required':bool(names),'request_count':len(requests)}
    start=time.perf_counter()
    try:
        rows=[];valid_strings=[]
        for req in requests:
            raw=req['identity']['raw_smiles'];mol=Chem.MolFromSmiles(raw) if raw.strip() else None
            valid=mol is not None and mol.GetNumAtoms()>0
            rows.append({'request_id':req['request_id'],'raw_smiles':raw,'valid':bool(valid),
                         'canonical_smiles':Chem.MolToSmiles(mol) if valid else '',
                         'invalid_reason':None if valid else 'empty_or_RDKit_parse_invalid'})
            if valid and raw not in valid_strings:valid_strings.append(raw)
        write(out/'recheck.json',rows);detail['unique_valid_strings']=len(valid_strings)
        byraw={}
        # DPQ has no ADMET import or model load. Invalid-only batches also skip model assembly.
        if names and valid_strings:
            import torch,pandas as pd
            from admet_ai import ADMETModel
            from admet_ai.constants import DEFAULT_MODELS_DIR
            torch.set_num_threads(8)
            if torch.cuda.is_available():raise RuntimeError('Predictor sees GPU')
            detail.update(torch=torch.__version__,pandas=pd.__version__,model_directory=str(DEFAULT_MODELS_DIR),
                          num_workers=0,fingerprint_multiprocessing_min=101,model_load_started=1)
            write(out/'model_load_started.json',detail);t=time.perf_counter()
            model=ADMETModel(models_dir=DEFAULT_MODELS_DIR,num_workers=0,fingerprint_multiprocessing_min=101)
            detail.update(model_load_completed=1,load_wall_seconds=time.perf_counter()-t,device=str(model.device))
            if str(model.device)!='cpu':raise RuntimeError('Predictor device mismatch')
            write(out/'model_loaded.json',detail);detail['predict_started']=1;write(out/'predict_started.json',detail)
            t=time.perf_counter();frame=model.predict(smiles=valid_strings)
            detail.update(predict_completed=1,predict_wall_seconds=time.perf_counter()-t)
            if not isinstance(frame,pd.DataFrame) or list(frame.index)!=valid_strings:raise ValueError('ADMET returned mismatched raw IDs/order')
            frame.to_csv(str(out/'admet_raw.csv'),index_label='smiles')
            if any(c not in frame.columns for c in names.values()):raise ValueError('Required ADMET column missing')
            for raw in valid_strings:byraw[raw]={name:finite_or_none(frame.loc[raw,col]) for name,col in names.items()}
        for row in rows:
            row['admet_raw_targets']=byraw.get(row['raw_smiles'],{})
            row['nonfinite_or_missing_fields']=[n for n in names if row['admet_raw_targets'].get(n) is None]
        write(out/'result.json',{'requests_sha256':sha(args.requests),'rows':rows,'identity':detail})
    except BaseException as exc:
        write(out/'exception.json',{'exception':repr(exc),'traceback':traceback.format_exc()});raise
    finally:
        detail['worker_wall_seconds']=time.perf_counter()-start;write(out/'costs.json',detail)


if __name__=='__main__':main()
