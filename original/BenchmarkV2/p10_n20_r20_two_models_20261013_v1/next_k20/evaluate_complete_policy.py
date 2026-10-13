"""Original GeLLMO metrics for audited complete sampled-policy terminals.

Uses frozen repeat 0 as the K1 control and all 20 terminal outputs for K20.
Requires the retained Python 3.6/RDKit 2020/pandas 1.0.5 chemistry environment.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TASKS = ['BDP', 'BDQ', 'BPQ', 'DPQ', 'BDPQ', 'MPQ', 'BDMQ', 'BHMQ', 'BMPQ', 'HMPQ']
COUNTS = dict(zip(TASKS, [500] * 7 + [118, 191, 96]))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def mean(xs):
    xs = list(xs)
    return math.fsum(xs) / len(xs) if xs else None


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scope', required=True, choices=['mistral_s1_properties', 'llama_icl_properties'])
    args = p.parse_args()
    root = HERE / args.scope
    out = root / 'metrics'
    out.mkdir(exist_ok=True)
    assert not (out / 'audit.json').exists(), 'Completed metrics are immutable'
    sources = {}

    def track(path):
        path = Path(path).resolve()
        sources[str(path)] = sha(path)
        return path

    def read(path):
        return json.loads(track(path).read_text())

    def lines(path):
        path = track(path)
        with (gzip.open(str(path), 'rt') if path.suffix == '.gz' else path.open()) as f:
            return [json.loads(s) for s in f if s.strip()]

    def write(name, value):
        (out / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

    def csvwrite(name, records):
        with (out / name).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(records[0]))
            w.writeheader()
            w.writerows(records)

    import numpy as np
    import pandas as pd
    from rdkit import Chem, DataStructs, RDLogger, rdBase
    from rdkit.Chem import AllChem
    sys.path.insert(0, str(REPO / 'data/props'))
    import sascorer
    RDLogger.DisableLog('rdApp.*')
    assert rdBase.rdkitVersion == '2020.03.3' and pd.__version__ == '1.0.5' and np.__version__ == '1.18.5'
    audit = read(root / 'reports/final_audit.json')
    assert audit['status'] == 'PASS_COMPLETE_TERMINAL_PROPERTIES'
    assert audit['inputs'] == 3905 and audit['terminal_slots'] == 78100
    assert audit['unknown_property_failures'] == 0 and audit['source_files_unchanged'] is True
    assert Path(audit['output_path']).resolve() == (root / 'results/terminal_inputs.jsonl.gz').resolve()
    assert sha(audit['output_path']) == audit['output_sha256']
    model, method = audit['model_id'], audit['method_id']
    assert model == ('gellmo_p6_mistral' if args.scope == 'mistral_s1_properties' else 'gellmo_p6_llama')
    records = lines(audit['output_path'])
    assert len(records) == 3905
    manifest_spec = read(REPO / 'BenchmarkV2/k1_nine_methods_execution_20261011_v2/config/manifests.json')
    manifests = {}
    for task in TASKS:
        spec = manifest_spec[task]
        assert sha(spec['path']) == spec['sha256']
        manifest = lines(spec['path'])
        assert len(manifest) == COUNTS[task]
        manifests[task] = {r['input_uid']: r for r in manifest}
    notebook = read(REPO / 'evaluate.ipynb')
    namespace = {}
    exec(compile(''.join(notebook['cells'][6]['source']), 'evaluate.ipynb:cell6', 'exec'), namespace)
    select = namespace['find_best_optimized']
    assert select({'qed': .1}, {'qed': [.2, .2]}, ['qed']) == 1
    novelty_reference = set(track(REPO / 'data/unique_mols_in_one_ds_pairs.txt').read_text().splitlines())
    for path in [Path(__file__), REPO / 'data/props/sascorer.py', REPO / 'data/props/fpscores.pkl.gz']:
        track(path)

    @lru_cache(maxsize=50000)
    def molecule(s):
        return Chem.MolFromSmiles(s) if s.strip() else None

    @lru_cache(maxsize=10000)
    def quality(src, dst):
        sm, dm = molecule(src), molecule(dst)
        assert sm is not None and dm is not None
        a = AllChem.GetMorganFingerprintAsBitVect(sm, 2, nBits=2048, useChirality=False)
        b = AllChem.GetMorganFingerprintAsBitVect(dm, 2, nBits=2048, useChirality=False)
        return float(DataStructs.TanimotoSimilarity(a, b)), float(sascorer.calculateScore(dm)), Chem.MolToSmiles(dm)

    results, seen, candidate_ids = [], set(), set()
    slot_count = 0
    for record in records:
        task, uid = record['task_id'], record['input_uid']
        assert (task, uid) not in seen
        seen.add((task, uid))
        assert record['model_id'] == model and record['method_id'] == method
        frozen = manifests[task][uid]
        for key in ['input_idx', 'source_smiles', 'source_properties']:
            assert record[key] == frozen[key], (task, uid, key)
        source = record['source_properties']
        candidates = record['candidates']
        assert len(candidates) == 20
        assert [c['original_slot_id'] for c in candidates] == list(range(20))
        eligible = []
        for c in candidates:
            assert c['candidate_id'] not in candidate_ids
            candidate_ids.add(c['candidate_id'])
            assert c['source_smiles'] == record['source_smiles']
            vals = c.get('properties') or {}
            valid = bool(c['rdkit_valid']) and molecule(c['raw_smiles']) is not None
            measurable = all(finite(vals.get(k)) for k in source)
            if valid and not measurable:
                assert c.get('known_molecular_property_failure'), ('unknown property gap', c['candidate_id'])
            eligible.append(valid and measurable)
        slot_count += 20
        for view, k in [('R1_K1_first_repeat', 1), ('R20_K20', 20)]:
            available = [c for c, ok in zip(candidates[:k], eligible[:k]) if ok]
            pool = {key: [c['properties'][key] for c in available] for key in source}
            idx = select(source, pool, list(source))
            success = idx is not None
            row = dict(model_id=model, method_id=method, task_id=task, input_uid=uid,
                       input_idx=record['input_idx'], readout=view, output_slots=k,
                       source_smiles=record['source_smiles'], source_properties=source,
                       success=success, n_eligible_slots=len(available),
                       valid_legacy=any(molecule(c['raw_smiles']) is not None for c in candidates[:k]),
                       valid_current=any(c['rdkit_valid'] and bool(c['raw_smiles'].strip()) for c in candidates[:k]),
                       representative_candidate_id=None, representative_repeat=None, representative_raw_smiles=None,
                       representative_canonical_smiles=None, representative_properties=None,
                       RI_official_code=None, Sim_raw=None, SAS_raw=None, Novelty_official=None,
                       APS={key: None for key in source})
            if success:
                c = available[idx]
                vals = c['properties']
                assert all(vals[k] < x if k == 'mutagenicity' else vals[k] > x for k, x in source.items())
                sim, sas, canon = quality(record['source_smiles'], c['raw_smiles'])
                raw = c.get('raw_properties') or {}
                for key, value in [('sim', sim), ('sas', sas)]:
                    if finite(raw.get(key)):
                        assert math.isclose(value, raw[key], rel_tol=1e-10, abs_tol=1e-10), (c['candidate_id'], key)
                row.update(representative_candidate_id=c['candidate_id'], representative_repeat=c['original_slot_id'],
                           representative_raw_smiles=c['raw_smiles'], representative_canonical_smiles=canon,
                           representative_properties={key: vals[key] for key in source},
                           RI_official_code=mean(abs(vals[key] - x) / abs(x) if x != 0 else (vals[key] - .1) / .1 for key, x in source.items()),
                           Sim_raw=sim, SAS_raw=sas, Novelty_official=canon not in novelty_reference,
                           APS={key: vals[key] for key in source})
            results.append(row)
        assert not results[-2]['success'] or results[-1]['success']
    assert slot_count == len(candidate_ids) == 78100 and len(results) == 7810
    assert seen == {(task, uid) for task, rows in manifests.items() for uid in rows}
    task_results, aps, macros = [], [], []
    fields = ['SR', 'Validity_legacy', 'Validity_current', 'RI_official_code', 'Sim_raw', 'SAS_raw', 'Novelty_official']
    for view in ['R1_K1_first_repeat', 'R20_K20']:
        for task in TASKS:
            rows = [r for r in results if r['task_id'] == task and r['readout'] == view]
            assert len(rows) == COUNTS[task]
            good = [r for r in rows if r['success']]
            row = dict(model_id=model, method_id=method, task_id=task, readout=view,
                       n_inputs=len(rows), n_success=len(good), SR=100. * len(good) / len(rows),
                       Validity_legacy=100. * mean(r['valid_legacy'] for r in rows),
                       Validity_current=100. * mean(r['valid_current'] for r in rows),
                       Novelty_official=100. * mean(r['Novelty_official'] for r in good) if good else None)
            row.update({key: mean(r[key] for r in good) for key in ['RI_official_code', 'Sim_raw', 'SAS_raw']})
            task_results.append(row)
            for prop in rows[0]['source_properties']:
                aps.append(dict(model_id=model, method_id=method, readout=view, task_id=task, property=prop,
                                n_success=len(good), APS=mean(r['APS'][prop] for r in good)))
        for scope, tasks in [('IND', TASKS[:5]), ('OOD', TASKS[5:]), ('Macro10', TASKS)]:
            rows = [r for r in task_results if r['readout'] == view and r['task_id'] in tasks]
            row = dict(model_id=model, method_id=method, readout=view, scope=scope,
                       n_tasks=len(rows), n_inputs=sum(r['n_inputs'] for r in rows))
            row.update({key: mean(r[key] for r in rows) if all(r[key] is not None for r in rows) else None for key in fields})
            macros.append(row)
    with gzip.open(str(out / 'per_input.jsonl.gz'), 'wt') as f:
        for row in results:
            f.write(json.dumps(row, allow_nan=False) + '\n')
    csvwrite('per_task.csv', task_results)
    csvwrite('macro.csv', macros)
    csvwrite('aps_by_task.csv', aps)
    write('metrics.json', dict(per_task=task_results, macro=macros, APS=aps))
    write('provenance.json', sources)
    assert all(sha(path) == digest for path, digest in sources.items())
    write('audit.json', dict(status='PASS_COMPLETE_SAMPLED_POLICY_R20_METRICS',
          generated_utc=datetime.utcnow().isoformat() + 'Z', model_id=model, method_id=method,
          inputs=3905, terminal_slots=78100, input_readouts=7810, task_groups=20,
          first_repeat_is_same_configuration_K1=True, old_greedy_or_beam_K1_not_mixed=True,
          all_source_hashes_unchanged=True, no_unknown_property_gaps=True,
          quality='Same original-selector successful representative for RI/Sim/SAS/Novelty/APS, stable repeat order.',
          K20_interpretation='At least one of twenty complete terminal outputs succeeds; evaluation coverage, not an inference selector.',
          versions=dict(rdkit=rdBase.rdkitVersion, numpy=np.__version__, pandas=pd.__version__),
          input_sha256=audit['output_sha256'], evaluator_sha256=sha(__file__)))
    print(json.dumps(macros, indent=2), flush=True)


if __name__ == '__main__':
    main()
