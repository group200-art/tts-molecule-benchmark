"""Extend the frozen source-cluster bootstrap to all 18 audited configurations."""
import csv
import gzip
import hashlib
import json
import math
import os
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
BENCH = ROOT.parent
PAPER = ROOT / 'paper'
DATA = PAPER / 'data'
FINAL = BENCH / 'benchmark_completion_20261023_v1/results'
TASKS = ['BDP', 'BDQ', 'BPQ', 'DPQ', 'BDPQ', 'MPQ', 'BDMQ', 'BHMQ', 'BMPQ', 'HMPQ']
MODELS = ['Mistral', 'Llama']
METHODS = ['beam', 'sampling', 'self_refine', 'tot', 'got', 's1', 'voting', 'icl', 'reflexion']
LABELS = dict(zip(METHODS, ['Beam', 'Sampling', 'Self-Refine', 'ToT', 'GoT', 's1', 'Exact Voting', '4-shot ICL', 'Reflexion']))
KS = [1, 5, 10, 20]
SOURCES = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def track(path):
    path = Path(path).resolve()
    if str(path) not in SOURCES:
        SOURCES[str(path)] = sha(path)
    return path


def read(path):
    return json.loads(track(path).read_text())


def rows(path):
    with track(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def jsonl(path):
    path = track(path)
    with (gzip.open if path.suffix == '.gz' else open)(path, 'rt') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def csvwrite(name, values):
    with (DATA / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Bootstrap computation requires a CPU allocation'
    complete = read(FINAL / 'completion.json')
    assert complete['status'] == 'COMPLETE_18_CONFIGURATIONS' and complete['missing_configurations'] == []
    for name, expected_sha in complete['output_sha256'].items():
        assert sha(name) == expected_sha, name
    expected = {}
    for task in TASKS:
        for record in jsonl(BENCH / 'manifests' / (task + '.test.seen.jsonl')):
            key = task, record['input_uid']
            assert key not in expected
            expected[key] = record
    assert len(expected) == 3905
    groups = defaultdict(dict)
    for row in jsonl(FINAL / 'per_input_quality_and_prefix.jsonl.gz'):
        config = row['model'], row['method_id']
        key = row['task_id'], row['input_uid']
        assert key in expected and key not in groups[config]
        assert type(row['success']) is bool and row['success'] == row['SR_K20']
        assert all(type(row['SR_K' + str(k)]) is bool for k in KS)
        assert [row['SR_K' + str(k)] for k in KS] == sorted(row['SR_K' + str(k)] for k in KS)
        groups[config][key] = row
    configs = [(model, method) for model in MODELS for method in METHODS]
    assert set(groups) == set(configs) and all(set(g) == set(expected) for g in groups.values())
    keys = list(expected)
    y = np.array([[groups[c][key]['success'] for c in configs] for key in keys], dtype=np.float64)
    task_idx = [np.array([i for i, key in enumerate(keys) if key[0] == task]) for task in TASKS]
    task_rates = np.array([100 * y[idx].mean(axis=0) for idx in task_idx])
    task_counts = np.array([y[idx].sum(axis=0).astype(int) for idx in task_idx])
    clusters = sorted({expected[key]['source_molecule_uid'] for key in keys})
    assert len(clusters) == 2860
    cluster_map = {name: i for i, name in enumerate(clusters)}
    cluster_idx = np.array([cluster_map[expected[key]['source_molecule_uid']] for key in keys])
    weights = np.random.default_rng(20261014).multinomial(
        len(clusters), np.full(len(clusters), 1 / len(clusters)), size=2000)
    old_audit = read(ROOT / 'baseline_data/table3_consistency_audit.json')
    weights_sha = hashlib.sha256(weights.astype('<i8').tobytes()).hexdigest()
    assert weights_sha == old_audit['cluster_weights_sha256_int64']
    boot_task = []
    for idx in task_idx:
        w = weights[:, cluster_idx[idx]]
        denominator = w.sum(axis=1)
        assert np.all(denominator > 0)
        boot_task.append(100 * (w @ y[idx]) / denominator[:, None])
    boot_task = np.array(boot_task)
    boot_macro = boot_task.mean(axis=0)
    ci = np.percentile(boot_macro, [2.5, 97.5], axis=0)
    macro, paired, consistency, per_task, prefix = [], [], [], [], []
    final_macro = {(r['model'], r['method_id']): r for r in rows(FINAL / 'macro.csv')}
    final_task = {(r['model'], r['method_id'], r['task_id']): r for r in rows(FINAL / 'per_task.csv')}
    for j, (model, method) in enumerate(configs):
        reference = configs.index((model, 'beam'))
        m = dict(model=model, method_id=method, method=LABELS[method],
                 construction='C20' if method in ['beam', 'sampling', 'tot', 'got'] else 'R20',
                 inputs=3905, output_slots=78100, IND=float(task_rates[:5, j].mean()),
                 OOD=float(task_rates[5:, j].mean()), Macro10=float(task_rates[:, j].mean()),
                 CI95_low=float(ci[0, j]), CI95_high=float(ci[1, j]), status='COMPLETE_EVALUATED_K20')
        assert math.isclose(m['Macro10'], float(final_macro[model, method]['SR20']), abs_tol=1e-10)
        macro.append(m)
        lo, hi = np.percentile(boot_macro[:, j] - boot_macro[:, reference], [2.5, 97.5])
        paired.append(dict(model=model, method_id=method, reference='beam',
            Macro10_difference_pp=float(task_rates[:, j].mean() - task_rates[:, reference].mean()),
            CI95_low_pp=float(lo), CI95_high_pp=float(hi),
            gained_task_inputs=int(((y[:, j] == 1) & (y[:, reference] == 0)).sum()),
            lost_task_inputs=int(((y[:, j] == 0) & (y[:, reference] == 1)).sum())))
        c = dict(model=model, method_id=method, method=LABELS[method], reference='beam', K=20)
        for domain, sl in [('IND', slice(0, 5)), ('OOD', slice(5, 10))]:
            domain_boot = boot_task[sl].mean(axis=0)
            lo, hi = np.percentile(domain_boot[:, j] - domain_boot[:, reference], [2.5, 97.5])
            c[domain + '_difference_pp'] = float(task_rates[sl, j].mean() - task_rates[sl, reference].mean())
            c[domain + '_CI95_low_pp'], c[domain + '_CI95_high_pp'] = float(lo), float(hi)
        differences = task_counts[:, j] - task_counts[:, reference]
        c.update(tasks_win=int((differences > 0).sum()), tasks_tie=int((differences == 0).sum()),
                 tasks_loss=int((differences < 0).sum()))
        exact = [Fraction(100 * int(differences[t]), len(task_idx[t])) for t in range(10)]
        c['minimum_task_difference_pp'] = float(min(exact))
        c['minimum_task_id'] = ';'.join(task for task, value in zip(TASKS, exact) if value == min(exact))
        c['status'] = 'COMPLETE_EVALUATED_K20'
        consistency.append(c)
        p = dict(model=model, method_id=method, method=LABELS[method], n_tasks=10, n_inputs=3905)
        for k in KS:
            p['K' + str(k)] = math.fsum(100 * sum(groups[model, method][keys[i]]['SR_K' + str(k)] for i in idx) / len(idx) for idx in task_idx) / 10
        assert math.isclose(p['K20'], m['Macro10'], abs_tol=1e-10)
        prefix.append(p)
        for t, task in enumerate(TASKS):
            f = final_task[model, method, task]
            assert int(f['n_success']) == int(task_counts[t, j])
            assert math.isclose(float(f['SR20']), task_rates[t, j], abs_tol=1e-10)
            per_task.append(dict(model=model, method_id=method, task_id=task,
                domain='IND' if t < 5 else 'OOD', n_inputs=len(task_idx[t]),
                n_success=int(task_counts[t, j]), SR20=float(task_rates[t, j])))
    # Preserve every previously published numeric statistic, including the
    # paired and task-group intervals, before adding any new configuration.
    checks = []
    for name, new_rows, fields in [
        ('macro_uncertainty.csv', macro, ['IND', 'OOD', 'Macro10', 'CI95_low', 'CI95_high']),
        ('paired_vs_beam.csv', paired, ['Macro10_difference_pp', 'CI95_low_pp', 'CI95_high_pp', 'gained_task_inputs', 'lost_task_inputs']),
        ('table3_k20_consistency.csv', consistency, ['IND_difference_pp', 'IND_CI95_low_pp', 'IND_CI95_high_pp', 'OOD_difference_pp', 'OOD_CI95_low_pp', 'OOD_CI95_high_pp', 'tasks_win', 'tasks_tie', 'tasks_loss', 'minimum_task_difference_pp']),
    ]:
        byconfig = {(r['model'], r['method_id']): r for r in new_rows}
        for old in rows(ROOT / 'baseline_data' / name):
            if old.get('status') not in (None, 'COMPLETE_EVALUATED_K20'):
                continue
            for field in fields:
                error = abs(float(byconfig[old['model'], old['method_id']][field]) - float(old[field]))
                assert error < 1e-10, (name, old['model'], old['method_id'], field, error)
                checks.append(error)
    for name, values in [('macro_uncertainty.csv', macro), ('paired_vs_beam.csv', paired),
                         ('table3_k20_consistency.csv', consistency), ('per_task.csv', per_task),
                         ('k20_prefix_macro.csv', prefix)]:
        csvwrite(name, values)
    csvwrite('k20_prefix_coverage.csv', rows(FINAL / 'prefix_coverage.csv'))
    quality_fields = ['model', 'method_id', 'n_inputs', 'n_slots', 'n_success', 'SR20', 'Valid_slots', 'SR_delta20', 'RI_01', 'Sim', 'SAS']
    csvwrite('table3_k20_quality.csv', [{k: final_macro[c][k] for k in quality_fields} for c in configs])
    csvwrite('table3_k20_quality_by_task.csv', rows(FINAL / 'per_task.csv'))
    main_rows = []
    macro_index = {(r['model'], r['method_id']): r for r in macro}
    for method in METHODS:
        row = dict(Method=LABELS[method], Output_construction=macro_index['Mistral', method]['construction'])
        for model in MODELS:
            for field in ['IND', 'OOD', 'Macro10', 'CI95_low', 'CI95_high', 'status']:
                row[model + '_' + field] = macro_index[model, method][field]
        main_rows.append(row)
    csvwrite('table1_k20.csv', main_rows)
    audit = dict(status='PASS_COMPLETE_18_CONFIGURATION_STATISTICS', configurations=18,
                 task_groups=180, task_inputs=70290, output_slots=1405800, source_clusters=2860,
                 bootstrap_replicates=2000, bootstrap_seed=20261014, numpy_version=np.__version__,
                 cluster_weights_sha256_int64=weights_sha, unchanged_old_statistics_checked=len(checks),
                 unchanged_old_statistics_max_error=max(checks), sources=SOURCES,
                 generation_calls=0, property_prediction_calls=0,
                 uncertainty_scope='Paired source-cluster resampling conditional on fixed saved generations; no seed variance or multiplicity adjustment',
                 derivation_script_sha256=sha(__file__))
    for name in ['statistics_audit.json', 'table3_consistency_audit.json']:
        (DATA / name).write_text(json.dumps(audit, indent=2) + '\n')
    (DATA / 'k20_data_audit.json').write_text(json.dumps(dict(status='PASS', configurations=18,
        task_groups=180, task_inputs=70290, output_slots=1405800, missing_configurations=[],
        final_completion_sha256=sha(FINAL / 'completion.json'), old_13_readouts_preserved_exactly=True,
        chemistry_or_model_inference=False, quality_precision='Retained comparison-round2 Sim and SAS',
        quality_aggregation='Within-task successful representative means, then equal ten-task macro'), indent=2) + '\n')
    print(json.dumps({k: v for k, v in audit.items() if k != 'sources'}, indent=2), flush=True)
    for row in paired:
        print(row, flush=True)


if __name__ == '__main__':
    main()
