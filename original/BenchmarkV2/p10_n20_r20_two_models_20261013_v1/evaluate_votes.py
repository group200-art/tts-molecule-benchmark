"""Evaluate frozen N20 voting endpoints, with N20/K1 and old N400/K1 controls.

CPU only; run in the retained GeLLMO Python 3.6 chemistry environment.
Voting decisions never consult any property values from this script.
"""
import csv
import gzip
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime
from functools import lru_cache
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
B = REPO / 'BenchmarkV2'
TASKS = ['BDP', 'BDQ', 'BPQ', 'DPQ', 'BDPQ', 'MPQ', 'BDMQ', 'BHMQ', 'BMPQ', 'HMPQ']
COUNTS = dict(zip(TASKS, [500] * 7 + [118, 191, 96]))
VIEWS = ['N20_R1_K1', 'N20_R20_K20', 'N400_R1_K1_reference']
SOURCES = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def track(path):
    path = Path(path).resolve()
    SOURCES[str(path)] = sha(path)
    return path


def read(path):
    return json.loads(track(path).read_text())


def readlines(path):
    path = track(path)
    with (gzip.open(str(path), 'rt') if path.suffix == '.gz' else path.open()) as f:
        return [json.loads(line) for line in f if line.strip()]


def save(name, value):
    (OUT / 'results' / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def csvsave(name, rows):
    with (OUT / 'results' / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def mean(values):
    values = list(values)
    return math.fsum(values) / len(values) if values else None


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def main():
    import numpy as np
    import pandas as pd
    from rdkit import Chem, DataStructs, RDLogger, rdBase
    from rdkit.Chem import AllChem
    sys.path.insert(0, str(REPO / 'data/props'))
    import sascorer
    RDLogger.DisableLog('rdApp.*')
    assert rdBase.rdkitVersion == '2020.03.3'
    assert pd.__version__ == '1.0.5' and np.__version__ == '1.18.5'
    (OUT / 'results').mkdir(exist_ok=True)
    contract = read(OUT / 'protocol.json')
    assert contract['N_votes_per_round'] == contract['R_rounds'] == contract['Kout'] == 20
    original_hashes = read(OUT / 'original_input_hashes.json')
    assert all(sha(path) == digest for path, digest in original_hashes.items())
    voting_audit = read(OUT / 'voting/audit.json')
    assert voting_audit['status'] == 'PASS_FULL_TWO_MODEL_N20_R20_WINNERS'
    assert voting_audit['total_terminal_slots'] == 156200
    for short in ['mistral', 'llama']:
        completion = read(OUT / 'voting' / ('completion_' + short + '.json'))
        assert completion['status'] == 'PASS' and completion['inputs'] == 3905
        assert sha(completion['output']['path']) == completion['output']['sha256']
    property_audit = read(OUT / 'properties/reports/final_audit.json')
    assert property_audit['status'] == 'PASS_ALL_FIXED_WINNER_PROPERTIES_COMPLETE'
    assert property_audit['unknown_property_failures'] == 0
    assert sha(property_audit['output_path']) == property_audit['output_sha256']
    notebook = read(REPO / 'evaluate.ipynb')
    namespace = {}
    exec(compile(''.join(notebook['cells'][6]['source']), 'evaluate.ipynb:cell6', 'exec'), namespace)
    original_select = namespace['find_best_optimized']
    assert original_select({'qed': .1}, {'qed': [.2, .2]}, ['qed']) == 1
    for path in [REPO / 'data/props/sascorer.py', REPO / 'data/props/fpscores.pkl.gz', Path(__file__)]:
        track(path)
    novelty_reference = set(track(REPO / 'data/unique_mols_in_one_ds_pairs.txt').read_text().splitlines())
    manifest_spec = read(B / 'k1_nine_methods_execution_20261011_v2/config/manifests.json')
    manifests = {}
    for task, spec in manifest_spec.items():
        assert sha(spec['path']) == spec['sha256']
        records = readlines(Path(spec['path']))
        assert len(records) == COUNTS[task]
        manifests[task] = {r['input_uid']: r for r in records}
    property_rows = readlines(OUT / 'properties/results/winner_properties.jsonl.gz')
    properties = {}
    for p in property_rows:
        key = tuple(p[k] for k in ['model_id', 'task_id', 'input_uid', 'raw_smiles'])
        assert key not in properties
        properties[key] = p

    @lru_cache(maxsize=50000)
    def molecule(smi):
        return Chem.MolFromSmiles(smi) if smi.strip() else None

    @lru_cache(maxsize=20000)
    def quality(source_smiles, output_smiles):
        src, mol = molecule(source_smiles), molecule(output_smiles)
        assert src is not None and mol is not None
        sfp = AllChem.GetMorganFingerprintAsBitVect(src, 2, nBits=2048, useChirality=False)
        ofp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useChirality=False)
        return (float(DataStructs.TanimotoSimilarity(sfp, ofp)),
                float(sascorer.calculateScore(mol)), Chem.MolToSmiles(mol))

    def eligible(p, source):
        vals = p.get('properties') or {}
        chemically_valid = p['rdkit_valid'] and molecule(p['raw_smiles']) is not None
        measurable = p.get('property_row_exists') is True and all(finite(vals.get(k)) for k in source)
        if chemically_valid and not measurable:
            assert p.get('known_molecular_property_failure'), ('Unclassified missing target properties', p)
        return chemically_valid and measurable

    def evaluate(model, record, view, outputs):
        source = record['source_properties']
        available = [(w, p) for w, p in outputs if eligible(p, source)]
        pool = {key: [p['properties'][key] for w, p in available] for key in source}
        index = original_select(source, pool, list(source))
        success = index is not None
        result = dict(model_id=model, task_id=record['task_id'], input_uid=record['input_uid'],
                      input_idx=record['input_idx'], readout=view, success=success,
                      output_slots=len(outputs), n_eligible_slots=len(available),
                      valid_legacy=any(molecule(p['raw_smiles']) is not None for w, p in outputs),
                      valid_current=any(p['rdkit_valid'] and bool(p['raw_smiles'].strip()) for w, p in outputs),
                      representative_candidate_id=None, representative_round_idx=None,
                      representative_raw_smiles=None, representative_canonical_smiles=None,
                      source_smiles=record['source_smiles'], source_properties=source,
                      representative_properties=None, RI_official_code=None, Sim_raw=None, SAS_raw=None,
                      Novelty_official=None, APS={key: None for key in source})
        if success:
            w, p = available[index]
            vals = p['properties']
            sim, sas, can = quality(record['source_smiles'], p['raw_smiles'])
            raw = p.get('raw_properties') or {}
            for key, value in [('sim', sim), ('sas', sas)]:
                if finite(raw.get(key)):
                    assert math.isclose(raw[key], value, rel_tol=1e-10, abs_tol=1e-10), (model, record['input_uid'], key)
            ri = mean(abs(vals[k] - x) / abs(x) if x != 0 else (vals[k] - .1) / .1 for k, x in source.items())
            result.update(representative_candidate_id=w['candidate_id'],
                          representative_round_idx=w.get('round_idx'), representative_raw_smiles=p['raw_smiles'],
                          representative_canonical_smiles=can, representative_properties={k: vals[k] for k in source},
                          RI_official_code=ri, Sim_raw=sim, SAS_raw=sas,
                          Novelty_official=can not in novelty_reference, APS={k: vals[k] for k in source})
        return result

    rows, task_results, aps, paired = [], [], [], []
    seen_inputs = set()
    n_winners = 0
    group_diagnostics = Counter()
    for short in ['mistral', 'llama']:
        model = 'gellmo_p6_' + short
        old = readlines(B / 'k1_nine_methods_execution_20261011_v2/results' / ('cached_' + short + '__ev.jsonl.gz'))
        old_by_key = {(r['task_id'], r['input_uid']): r for r in old}
        assert len(old_by_key) == 3905
        grouped = readlines(OUT / 'voting' / ('grouped_inputs_' + short + '.jsonl.gz'))
        assert len(grouped) == 3905
        for g in grouped:
            task, uid = g['task_id'], g['input_uid']
            key = (model, task, uid)
            assert key not in seen_inputs
            seen_inputs.add(key)
            record = manifests[task][uid]
            assert g['model_id'] == model and g['input_idx'] == record['input_idx']
            assert g['source_properties'] == record['source_properties']
            assert g['source_smiles'] == record['source_smiles']
            assert [w['round_idx'] for w in g['winners']] == list(range(20))
            outputs = []
            for w in g['winners']:
                assert 20 * w['round_idx'] <= w['original_slot_id'] < 20 * (w['round_idx'] + 1)
                p = properties[model, task, uid, w['raw_smiles']]
                assert p['rdkit_valid'] == w['rdkit_valid_ev']
                if 'source_smiles' in p:
                    assert p['source_smiles'] == record['source_smiles']
                if 'source_properties' in p:
                    assert p['source_properties'] == record['source_properties']
                outputs.append((w, p))
                group_diagnostics[short + '_top_vote_one'] += int(w['top_votes'] == 1)
                group_diagnostics[short + '_tied_maximum'] += int(w['tie_count'] > 1)
            n_winners += len(outputs)
            reference = old_by_key[task, uid]
            original = reference['selected_candidate']
            results = [evaluate(model, record, VIEWS[0], outputs[:1]),
                       evaluate(model, record, VIEWS[1], outputs),
                       evaluate(model, record, VIEWS[2], [(dict(candidate_id=original['candidate_id']), original)])]
            assert not results[0]['success'] or results[1]['success']
            assert results[2]['success'] == reference['evaluation']['SR_dir_success']
            rows.extend(results)
            paired.append(dict(model_id=model, task_id=task, input_uid=uid,
                               N20_K1_success=results[0]['success'], N20_K20_success=results[1]['success'],
                               N400_K1_success=results[2]['success']))
        print('evaluated', short, '3905 inputs', flush=True)
    assert len(seen_inputs) == 7810 and n_winners == 156200 and len(rows) == 23430
    for model in ['gellmo_p6_mistral', 'gellmo_p6_llama']:
        for view in VIEWS:
            for task in TASKS:
                group = [r for r in rows if (r['model_id'], r['readout'], r['task_id']) == (model, view, task)]
                assert len(group) == COUNTS[task]
                good = [r for r in group if r['success']]
                result = dict(model_id=model, readout=view, task_id=task, n_inputs=len(group), n_success=len(good),
                              SR=100. * len(good) / len(group),
                              Validity_legacy=100. * mean(r['valid_legacy'] for r in group),
                              Validity_current=100. * mean(r['valid_current'] for r in group))
                result.update({key: mean(r[key] for r in good) for key in ['RI_official_code', 'Sim_raw', 'SAS_raw']})
                result['Novelty_official'] = 100. * mean(r['Novelty_official'] for r in good) if good else None
                task_results.append(result)
                for prop in group[0]['source_properties']:
                    aps.append(dict(model_id=model, readout=view, task_id=task, property=prop,
                                    n_success=len(good), APS=mean(r['APS'][prop] for r in good)))
    macros = []
    for model in ['gellmo_p6_mistral', 'gellmo_p6_llama']:
        for view in VIEWS:
            for scope, tasks in [('IND', TASKS[:5]), ('OOD', TASKS[5:]), ('Macro10', TASKS)]:
                group = [r for r in task_results if r['model_id'] == model and r['readout'] == view and r['task_id'] in tasks]
                assert len(group) == len(tasks)
                result = dict(model_id=model, readout=view, scope=scope, n_tasks=len(tasks), n_inputs=sum(r['n_inputs'] for r in group))
                for metric in ['SR', 'Validity_legacy', 'Validity_current', 'RI_official_code', 'Sim_raw', 'SAS_raw', 'Novelty_official']:
                    result[metric] = mean(r[metric] for r in group) if all(r[metric] is not None for r in group) else None
                macros.append(result)
    historical = read(B / 'k20_nine_full_20261003_v1/p10_comparison/final_comparison.json')
    checks = {'N20_R1_K1': 'first20_K1_SR', 'N20_R20_K20': 'new20x20_K20_SR', 'N400_R1_K1_reference': 'old400_K1_SR'}
    for row in macros:
        if row['model_id'] == 'gellmo_p6_mistral' and row['scope'] == 'Macro10':
            assert math.isclose(row['SR'], historical['macro10'][checks[row['readout']]], abs_tol=1e-10)
    for name, records in [('per_input.jsonl.gz', rows), ('paired_outcomes.jsonl.gz', paired)]:
        with gzip.open(str(OUT / 'results' / name), 'wt') as f:
            for row in records:
                f.write(json.dumps(row, allow_nan=False) + '\n')
    csvsave('per_task.csv', task_results)
    csvsave('macro.csv', macros)
    csvsave('aps_by_task.csv', aps)
    save('metrics.json', dict(per_task=task_results, macro=macros, APS=aps))
    save('provenance.json', SOURCES)
    assert all(sha(path) == digest for path, digest in SOURCES.items())
    assert all(sha(path) == digest for path, digest in original_hashes.items())
    save('audit.json', dict(status='PASS_COMPLETE_TWO_MODELS_FULL3905_N20_R20_K20',
        generated_utc=datetime.utcnow().isoformat() + 'Z', models=2, inputs_per_model=3905,
        voting_endpoints=n_winners, per_input_readouts=len(rows), task_groups=len(task_results),
        new_LLM_generations=0, all_old_N400_success_labels_unchanged=True,
        Mistral_historical_N20_results_match=True, first_group_is_paired_K1=True,
        duplicate_and_invalid_winners_retained=True, group_diagnostics=dict(group_diagnostics),
        quality='Same representative for RI/Sim/SAS/Novelty/APS; original code selector on stable round order.',
        scope='N20 budget variant with 20 complete vote endpoints; not 20 N400 runs and not the original bootstrap scaling experiment.',
        source_files_unchanged=True, original_input_hashes_unchanged=True,
        script_sha256=sha(__file__)))
    assert all(sha(path) == digest for path, digest in SOURCES.items())
    print(json.dumps([r for r in macros if r['scope'] == 'Macro10'], indent=2), flush=True)


if __name__ == '__main__':
    main()
