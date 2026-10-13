"""DPQ chemistry recheck only: no ADMET import, assembly or prediction."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from contracts import read,write,sha
from runtime_common import require_allocation

def main():
    import argparse,os,platform,time
    p=argparse.ArgumentParser();p.add_argument('--requests',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();require_allocation()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU recheck must hide GPU')
    data=read(args.requests)
    if data['task_id']!='DPQ' or data['admet_columns']!={}:raise ValueError('Only registered no-ADMET task')
    from rdkit import Chem,rdBase
    start=time.perf_counter();rows=[];out=Path(args.output);out.mkdir(exist_ok=False)
    for req in data['requests']:
        raw=req['identity']['raw_smiles'];mol=Chem.MolFromSmiles(raw) if raw.strip() else None
        valid=mol is not None and mol.GetNumAtoms()>0
        rows.append(dict(request_id=req['request_id'],raw_smiles=raw,valid=bool(valid),canonical_smiles=Chem.MolToSmiles(mol) if valid else '',
                         invalid_reason=None if valid else 'empty_or_RDKit_parse_invalid',admet_raw_targets={},nonfinite_or_missing_fields=[]))
    write(out/'recheck.json',rows)
    write(out/'chemistry_result.json',dict(requests_sha256=sha(args.requests),rows=rows,identity=dict(interpreter=sys.executable,python=platform.python_version(),
        rdkit=rdBase.rdkitVersion,device='cpu',admet_status='not_required',model_load_calls=0,predict_started=0,predict_completed=0,
        load_wall_seconds=None,predict_wall_seconds=None,recheck_wall_seconds=time.perf_counter()-start)))
if __name__=='__main__':main()
