"""CPU-only diagnosis; reports errors without changing candidate values or validity."""
import argparse
import json
import os
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['inf', 'standard'], required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    root = Path(__file__).resolve().parent
    path = root / 'runs/s1_BDQ_0240_0320/s1_BDQ_0240_0320_b16/terminals/requests.json'
    data = json.loads(path.read_text())
    from rdkit import Chem, rdBase
    from rdkit.Chem import QED
    functions = dict(qed=lambda raw: QED.qed(Chem.MolFromSmiles(raw)),
                     remove_hs=lambda raw: Chem.MolToSmiles(Chem.RemoveHs(Chem.MolFromSmiles(raw))))
    if args.mode == 'standard':
        sys.path.insert(0, str(root.parents[1] / 'data/props'))
        import properties
        functions.update(standard_qed=properties.qed, standard_drd2=properties.drd2,
                         standard_sas=properties.sas)
    rows = []
    for request in data['requests']:
        raw = request['identity']['raw_smiles']
        mol = Chem.MolFromSmiles(raw) if raw.strip() else None
        row = dict(request_id=request['request_id'], raw_smiles=raw,
                   parse_valid=mol is not None and mol.GetNumAtoms() > 0, operations={})
        if row['parse_valid']:
            for name, fn in functions.items():
                try:
                    value = fn(raw)
                    row['operations'][name] = dict(status='ok', value=value)
                except Exception as err:
                    row['operations'][name] = dict(status='error', type=type(err).__name__, error=str(err))
        rows.append(row)
    result = dict(mode=args.mode, python=sys.version, rdkit=rdBase.rdkitVersion, job=os.environ['SLURM_JOB_ID'],
                  requests=str(path), rows=rows, scientific_outputs_changed=False)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as f:
        json.dump(result, f, indent=2)
    errors = [r for r in rows if any(v['status'] == 'error' for v in r['operations'].values())]
    print(json.dumps(dict(mode=args.mode, inputs=len(rows), parse_valid=sum(r['parse_valid'] for r in rows),
                          errors=errors)), flush=True)


if __name__ == '__main__':
    main()
