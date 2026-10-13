"""Reevaluate all K1 endpoints using the unchanged registered metric functions.

No model, chemistry package, cluster path, network, or GPU is required.
Cached target properties and recorded chemical eligibility are inputs, not recomputed.
"""
import ast
import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from names import renamed_row

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / 'original/BenchmarkV2'


def read(path):
    return json.loads(path.read_text())


def load_evaluator():
    kernel = {}
    source = ORIGINAL / 'src/metrics_cpu.py'
    exec(compile(source.read_text(), str(source), 'exec'), kernel)
    tasks = {r['task_id']: r for r in read(ORIGINAL / 'registries/task_registry.draft.json')['tasks']}

    def task_contract(task_id, supplied=None):
        task = tasks[task_id]
        if supplied is not None and supplied != task['objectives']:
            raise ValueError('Task objectives differ')
        return task

    def finite_or_none(value):
        try:
            x = float(value)
            return x if not isinstance(value, bool) and math.isfinite(x) else None
        except (TypeError, ValueError, OverflowError):
            return None

    source = ORIGINAL / 'next3_ten_tasks_run01_v1/src/metrics_entry.py'
    tree = ast.parse(source.read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('evaluate_registered', 'aggregate_registered')]
    assert len(tree.body) == 2
    namespace = dict(kernel=lambda: kernel, task_contract=task_contract, finite_or_none=finite_or_none)
    exec(compile(ast.fix_missing_locations(tree), str(source), 'exec'), namespace)
    return namespace['evaluate_registered'], namespace['aggregate_registered']


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    from verify_snapshot import main as verify
    verify()
    evaluate, aggregate = load_evaluator()
    cfg = read(ORIGINAL / 'k1_nine_methods_execution_20261011_v2/config/methods.json')
    expected_audit = read(ROOT / 'original/results/k1/endpoint_audit.json')
    assert expected_audit['status'] == 'COMPLETE_FULL3905_TWO_MODELS_NINE_METHODS_K1'
    manifests = {}
    for task, spec in read(ROOT / 'data/manifests/index.json').items():
        blob = (ROOT / spec['path']).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == spec['sha256']
        records = [json.loads(s) for s in blob.splitlines() if s.strip()]
        manifests[task] = {r['input_uid']: r for r in records}
        assert len(records) == len(manifests[task]) == spec['inputs']
    assert sum(map(len, manifests.values())) == 3905
    approved_exceptions = {
        (r['model_id'], r['method_id'], r['task_id'], r['input_uid'])
        for r in expected_audit['accepted_property_failures']}
    groups = defaultdict(dict)
    expected_files = {Path(p).name for p in expected_audit['source_files']}
    actual_files = {p.name for p in (ROOT / 'data/k1_endpoints').glob('*.jsonl.gz')}
    assert actual_files == expected_files and len(actual_files) == 18
    for filename in sorted(actual_files):
        with gzip.open(str(ROOT / 'data/k1_endpoints' / filename), 'rt') as f:
            for line in f:
                r = json.loads(line)
                model, method, task, uid = (r[k] for k in ('model_id', 'method_id', 'task_id', 'input_uid'))
                assert model in cfg['models'] and method in cfg['methods'] and task in cfg['tasks']
                frozen = manifests[task][uid]
                assert r['source_properties'] == frozen['source_properties'] and r['input_idx'] == frozen['input_idx']
                group = groups[model, method, task]
                assert uid not in group
                candidate = r['selected_candidate']
                assert candidate['original_slot_id'] == 0
                assert type(candidate['rdkit_valid']) is bool
                props = candidate.get('properties')
                finite = isinstance(props, dict) and all(
                    isinstance(props.get(k), (int, float)) and not isinstance(props[k], bool)
                    and math.isfinite(props[k]) for k in r['source_properties'])
                if candidate['rdkit_valid'] and candidate['raw_smiles'].strip() and not finite:
                    assert (model, method, task, uid) in approved_exceptions, 'Unclassified property gap'
                computed = evaluate(task, uid, r['source_properties'], [candidate], 1)
                for key in ('SR_dir_success', 'SR_delta_success', 'n_slots', 'RI_legacy', 'RI_abs_eps01',
                            'selected_candidate_id_dir', 'selected_smiles_dir', 'input_valid'):
                    assert computed[key] == r['evaluation'][key], (model, method, task, uid, key)
                group[uid] = computed
    assert len(groups) == 180
    task_rows, table_rows = [], []
    for method in cfg['methods']:
        table_row = dict(method_id=method)
        for model in cfg['models']:
            rates = {}
            for task in cfg['tasks']:
                group = groups[model, method, task]
                assert set(group) == set(manifests[task])
                rows = [group[uid] for uid in manifests[task]]
                metrics = aggregate(rows)
                rates[task] = 100.0 * sum(r['SR_dir_success'] for r in rows) / len(rows)
                task_rows.append(dict(model_id=model, method_id=method, task_id=task, n_inputs=len(rows),
                                      SR=rates[task], SR_delta=100.0 * sum(r['SR_delta_success'] for r in rows) / len(rows),
                                      Validity=100.0 * sum(r['input_valid'] for r in rows) / len(rows)))
            for scope, tasks in [('IND', cfg['tasks'][:5]), ('OOD', cfg['tasks'][5:]), ('Macro10', cfg['tasks'])]:
                table_row[model + '_' + scope] = sum(rates[t] for t in tasks) / len(tasks)
        table_rows.append(table_row)
    expected = read(ROOT / 'original/results/k1/table1.json')
    assert len(table_rows) == len(expected)
    for row, reference in zip(table_rows, expected):
        assert row['method_id'] == reference['method_id']
        for key, value in row.items():
            if key != 'method_id':
                assert math.isclose(value, reference[key], rel_tol=1e-12, abs_tol=1e-10), key
    count = sum(len(g) for g in groups.values())
    assert count == 70290
    output = ROOT / 'analysis_output'
    output.mkdir(exist_ok=True)
    write_csv(output / 'table1.csv', [renamed_row(row) for row in table_rows])
    write_csv(output / 'task_metrics.csv', [renamed_row(row) for row in task_rows])
    audit = dict(status='PASS', endpoint_records=count, configurations=18, task_groups=180,
                 recomputed_SR_and_delta_and_RI_match=True, table1_matches=True,
                 property_values='Saved original precision; not new property predictions.')
    (output / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print('PASS: all 70,290 endpoints and all Table 1 values match.')
    print('Results: analysis_output/table1.csv and analysis_output/task_metrics.csv')


if __name__ == '__main__':
    main()
