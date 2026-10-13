"""Regression-check the generic metric evaluator against completed real P10 data.

All fixture inputs and generated test metrics stay in a marked temporary folder.
They are never accepted as s1 or ICL experimental results.
"""
import gzip
import hashlib
import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
P10 = HERE.parent


def rows(path):
    with gzip.open(str(path), 'rt') as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    assert json.loads((P10 / 'results/audit.json').read_text())['status'] == 'PASS_COMPLETE_TWO_MODELS_FULL3905_N20_R20_K20'
    testroot = Path(tempfile.mkdtemp(prefix='gellmo_metrics_regression_NOT_EXPERIMENT_'))
    props = {(p['model_id'], p['task_id'], p['input_uid'], p['raw_smiles']): p
             for p in rows(P10 / 'properties/results/winner_properties.jsonl.gz')}
    reference = json.loads((P10 / 'results/metrics.json').read_text())
    checked = 0
    for short, scope in [('mistral', 'mistral_s1_properties'), ('llama', 'llama_icl_properties')]:
        root = testroot / scope
        (root / 'reports').mkdir(parents=True)
        (root / 'results').mkdir()
        converted = []
        for g in rows(P10 / 'voting' / ('grouped_inputs_' + short + '.jsonl.gz')):
            record = {k: g[k] for k in ['model_id', 'task_id', 'input_uid', 'input_idx', 'source_smiles', 'source_properties']}
            record['method_id'] = 'P10_REGRESSION_FIXTURE_NOT_S1_OR_ICL'
            record['candidates'] = []
            for w in g['winners']:
                p = dict(props[g['model_id'], g['task_id'], g['input_uid'], w['raw_smiles']])
                p.update(candidate_id=w['candidate_id'], original_slot_id=w['round_idx'], source_smiles=g['source_smiles'])
                record['candidates'].append(p)
            converted.append(record)
        target = root / 'results/terminal_inputs.jsonl.gz'
        with gzip.open(str(target), 'wt') as f:
            for r in converted:
                f.write(json.dumps(r, allow_nan=False) + '\n')
        audit = dict(status='PASS_COMPLETE_TERMINAL_PROPERTIES', inputs=3905, terminal_slots=78100,
                     unknown_property_failures=0, source_files_unchanged=True, model_id='gellmo_p6_' + short,
                     method_id='P10_REGRESSION_FIXTURE_NOT_S1_OR_ICL', output_path=str(target),
                     output_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                     fixture_only=True, fixture_source=str(P10))
        (root / 'reports/final_audit.json').write_text(json.dumps(audit))
        spec = importlib.util.spec_from_file_location('tested_policy_metrics', str(HERE / 'evaluate_complete_policy.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.HERE = testroot
        sys.argv = [str(HERE / 'evaluate_complete_policy.py'), '--scope', scope]
        module.main()
        got = json.loads((root / 'metrics/metrics.json').read_text())
        names = {'R1_K1_first_repeat': 'N20_R1_K1', 'R20_K20': 'N20_R20_K20'}
        for kind in ['per_task', 'macro', 'APS']:
            keys = ['model_id', 'readout'] + (['scope'] if kind == 'macro' else ['task_id']) + (['property'] if kind == 'APS' else [])
            indexed = {tuple(r[k] for k in keys): r for r in reference[kind]}
            for r in got[kind]:
                key = tuple(names[r[k]] if k == 'readout' else r[k] for k in keys)
                expected = indexed[key]
                for field, value in expected.items():
                    if field == 'readout':
                        assert names[r[field]] == value
                    elif isinstance(value, (int, float)):
                        assert math.isclose(r[field], value, abs_tol=1e-10, rel_tol=1e-12), (short, kind, key, field)
                    else:
                        assert r[field] == value, (short, kind, key, field)
                    checked += 1
    report = dict(status='PASS_REAL_DATA_METRIC_REGRESSION', fixture_only=True, fixture_root=str(testroot),
                  compared_models=2, compared_input_readouts=15620, checked_aggregate_fields=checked,
                  basis='Two-model N20 voting data, two readouts, previously independently audited; no s1/ICL result claims.',
                  evaluator_sha256=hashlib.sha256((HERE / 'evaluate_complete_policy.py').read_bytes()).hexdigest())
    out = HERE / 'independent_review'
    out.mkdir(exist_ok=True)
    (out / 'generic_metrics_regression.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
