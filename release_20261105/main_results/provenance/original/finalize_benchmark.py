"""Complete the frozen 13-configuration data snapshot with five audited results.

No generation, property prediction, model selection, or manuscript write occurs.
The accepted manuscript's exact per-candidate readout functions are reused.
"""
import ast
import csv
import fcntl
import gzip
import hashlib
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent
B = ROOT.parent
BASE = B / 'recomb_k20_three_tables_20261018_v1'
TASKS = ['BDP', 'BDQ', 'BPQ', 'DPQ', 'BDPQ', 'MPQ', 'BDMQ', 'BHMQ', 'BMPQ', 'HMPQ']
METHODS = ['beam', 'sampling', 'self_refine', 'tot', 'got', 's1', 'voting', 'icl', 'reflexion']
MODELS = ['Mistral', 'Llama']
KS = [1, 5, 10, 20]
NEW = [
    ('Llama', 'self_refine', ROOT / 'llama_self_refine_properties',
     'reports/completion.json', 'PASS_COMPLETE_LLAMA_SELF_REFINE_K1_K20', 'R20_K20'),
    ('Mistral', 'tot', ROOT / 'tot/properties/mistral',
     'reports/paired_audit.json', 'PASS_COMPLETE_PAIRED_TOT_C20', 'C20_K20'),
    ('Llama', 'tot', ROOT / 'tot/properties/llama',
     'reports/paired_audit.json', 'PASS_COMPLETE_PAIRED_TOT_C20', 'C20_K20'),
    ('Mistral', 'reflexion', ROOT / 'mistral_reflexion_evaluation',
     'reports/completion.json', 'FULL_FINAL', 'R20_K20'),
    ('Llama', 'reflexion', ROOT / 'llama_reflexion',
     'reports/completion.json', 'FULL_FINAL', 'R20_K20'),
]
SOURCES = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def track(path):
    path = Path(path).resolve()
    SOURCES[str(path)] = sha(path)
    return path


def read(path):
    return json.loads(track(path).read_text())


def lines(path):
    path = track(path)
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def csvwrite(path, rows):
    tmp = path.with_name('.' + path.name + '.' + str(os.getpid()) + '.tmp')
    try:
        with tmp.open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(str(tmp), str(path))
    finally:
        if tmp.exists():
            tmp.unlink()


def mean(values):
    values = list(values)
    return math.fsum(values) / len(values) if values and all(v is not None for v in values) else None


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Final data audit runs in a CPU allocation'
    out = ROOT / 'results'
    out.mkdir(exist_ok=True)
    lock = (out / '.finalize.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    track(Path(__file__))
    completion = out / 'completion.json'
    if completion.exists():
        previous = json.loads(completion.read_text())
        assert previous['status'] == 'COMPLETE_18_CONFIGURATIONS'
        for path, expected in previous['output_sha256'].items():
            assert sha(path) == expected, path
        print('Previously completed and verified: 18 configurations')
        return
    frozen = read(ROOT / 'baseline_freeze.json')
    for path, expected in frozen['sha256'].items():
        assert sha(path) == expected, 'Accepted baseline changed: ' + path
    expected = {}
    for task in TASKS:
        for record in lines(B / 'manifests' / (task + '.test.seen.jsonl')):
            expected[task, record['input_uid']] = record
    assert len(expected) == 3905
    groups = defaultdict(dict)
    baseline = read(BASE / 'data_audit.json')
    assert baseline['status'] == 'PASS' and baseline['configurations'] == 13
    old_rows = lines(BASE / 'per_input_quality_and_prefix.jsonl.gz')
    for row in old_rows:
        key = row['task_id'], row['input_uid']
        group = row['model'], row['method_id']
        assert key in expected and key not in groups[group]
        groups[group][key] = {k: v for k, v in row.items() if k not in ('model', 'method_id')}
    assert len(old_rows) == 50765 and len(groups) == 13
    assert not any((model, method) in groups for model, method, *_ in NEW)
    algorithm_path = track(BASE / 'build_data.py')
    tree = ast.parse(algorithm_path.read_text())
    names = {'finite', 'mean', 'close', 'quality', 'add'}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == names
    namespace = dict(math=math, Decimal=Decimal, KS=KS, EXPECTED=expected, GROUPS=groups,
                     DELTAS=dict(bbbp=.2, drd2=.2, plogp=1., qed=.1, hia=.1, mutagenicity=.1))
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(algorithm_path), 'exec'), namespace)
    configurations = []
    for model, method, directory, marker, required_status, readout in NEW:
        receipt = read(directory / marker)
        assert receipt['status'] == required_status, (directory, receipt['status'])
        audit = read(directory / 'reports/final_audit.json')
        assert audit['status'] == 'PASS_COMPLETE_TERMINAL_PROPERTIES'
        assert audit['inputs'] == 3905 and audit['terminal_slots'] == 78100
        assert audit['unknown_property_failures'] == 0 and audit['source_files_unchanged']
        assert audit['model_id'] == 'gellmo_p6_' + model.lower()
        expected_method = ('p5_self_refine_r1_k20_p6_v1' if method == 'self_refine' else
                           'tot_lm_vote_b4_c20' if method == 'tot' else
                           'Reflexion-3trials-sampled-R20' if model == 'Mistral' else
                           'reflexion-max3-sampled-R20')
        assert audit['method_id'] == expected_method
        inputs_path = directory / 'results/terminal_inputs.jsonl.gz'
        assert sha(inputs_path) == audit['output_sha256']
        metrics_audit = read(directory / 'metrics/audit.json')
        assert metrics_audit['input_sha256'] == audit['output_sha256']
        archive = {(row['task_id'], row['input_uid']): row
                   for row in lines(directory / 'metrics/per_input.jsonl.gz') if row['readout'] == readout}
        assert set(archive) == set(expected)
        records = lines(inputs_path)
        assert len(records) == 3905
        for record in records:
            namespace['add'](model, method, record, record['candidates'],
                             archive[record['task_id'], record['input_uid']])
        configurations.append(dict(model=model, method_id=method, completion_record=str(directory / marker),
                                   terminal_inputs=str(inputs_path), metrics=str(directory / 'metrics/metrics.json')))
    assert set(groups) == {(model, method) for model in MODELS for method in METHODS}
    compact, per_task, macro, prefixes = [], [], [], []
    for model in MODELS:
        for method in METHODS:
            inputs = groups[model, method]
            assert set(inputs) == set(expected)
            task_rows = []
            for task in TASKS:
                rows = [inputs[key] for key in expected if key[0] == task]
                good = [row for row in rows if row['success']]
                result = dict(model=model, method_id=method, task_id=task, n_inputs=len(rows),
                              n_success=len(good), n_delta=sum(row['delta'] for row in rows),
                              n_valid_slots=sum(row['n_valid_slots'] for row in rows),
                              SR20=100 * len(good) / len(rows),
                              Valid_slots=100 * sum(row['n_valid_slots'] for row in rows) / (20 * len(rows)),
                              SR_delta20=100 * sum(row['delta'] for row in rows) / len(rows),
                              **{key: mean(row[key] for row in good) for key in ('RI_01', 'Sim', 'SAS')})
                per_task.append(result)
                task_rows.append(result)
                for k in KS:
                    n_success = sum(row['SR_K' + str(k)] for row in rows)
                    prefixes.append(dict(model=model, method_id=method, task_id=task, K=k,
                                         n_inputs=len(rows), n_success=n_success, SR=100 * n_success / len(rows)))
                for row in rows:
                    assert all(not row['SR_K' + str(a)] or row['SR_K' + str(b)] for a, b in zip(KS, KS[1:]))
                    compact.append(dict(model=model, method_id=method, **row))
            macro.append(dict(model=model, method_id=method, n_inputs=3905, n_slots=78100,
                              n_success=sum(row['n_success'] for row in task_rows),
                              IND=mean(row['SR20'] for row in task_rows[:5]),
                              OOD=mean(row['SR20'] for row in task_rows[5:]),
                              **{key: mean(row[key] for row in task_rows)
                                 for key in ('SR20', 'Valid_slots', 'SR_delta20', 'RI_01', 'Sim', 'SAS')},
                              tasks_with_success=sum(row['n_success'] > 0 for row in task_rows)))
    assert len(compact) == 70290 and len(per_task) == 180 and len(macro) == 18
    outputs = {}
    for name, rows in [('per_task.csv', per_task), ('macro.csv', macro), ('prefix_coverage.csv', prefixes)]:
        path = out / name
        if path.exists():
            # A prior incomplete CPU aggregation may have published exact CSVs.
            # Keep them for review; do not silently overwrite differing content.
            actual = list(csv.DictReader(path.open()))
            rendered = [{key: '' if value is None else str(value) for key, value in row.items()} for row in rows]
            assert actual == rendered, path
        else:
            csvwrite(path, rows)
        outputs[str(path)] = sha(path)
    path = out / 'per_input_quality_and_prefix.jsonl.gz'
    if path.exists():
        assert lines(path) == compact
    else:
        tmp = path.with_name('.' + path.name + '.' + str(os.getpid()) + '.tmp')
        try:
            with tmp.open('xb') as stream:
                with gzip.GzipFile(fileobj=stream, mode='wb', mtime=0) as compressed:
                    for row in compact:
                        compressed.write((json.dumps(row, allow_nan=False) + '\n').encode())
                stream.flush()
                os.fsync(stream.fileno())
            os.link(str(tmp), str(path))
        finally:
            if tmp.exists():
                tmp.unlink()
    outputs[str(path)] = sha(path)
    for path, expected_hash in SOURCES.items():
        assert sha(path) == expected_hash, path
    receipt = dict(status='COMPLETE_18_CONFIGURATIONS', completed_utc=datetime.now(timezone.utc).isoformat(),
                   configurations=18, new_configurations=5, task_groups=180, task_inputs=70290,
                   output_slots=1405800, missing_configurations=[], unknown_property_failures=0,
                   old_13_readouts_preserved_exactly=True, new_results=configurations,
                   quality_aggregation='Equal task means; undefined if any task has zero successful representatives',
                   uncertainty_intervals_recomputed=False, manuscript_or_overleaf_modified=False,
                   source_sha256=SOURCES, output_sha256=outputs)
    tmp = completion.with_name('.completion.' + str(os.getpid()) + '.tmp')
    try:
        with tmp.open('x') as stream:
            json.dump(receipt, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(str(tmp), str(completion))
    finally:
        if tmp.exists():
            tmp.unlink()
    print('PASS: all 18 configurations, 70,290 task-inputs, 1,405,800 fixed output slots')


if __name__ == '__main__':
    main()
