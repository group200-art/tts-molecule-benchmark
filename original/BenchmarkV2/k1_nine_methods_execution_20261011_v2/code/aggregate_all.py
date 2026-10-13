"""Audit real K1 endpoints and publish Table1 only when all18 configs are complete."""
import argparse
import csv
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
from cache_metrics import evaluate_registered, aggregate_registered
from cache_metrics import KERNEL
from export_cached import known_failure

MODELS = ['gellmo_p6_mistral', 'gellmo_p6_llama']
METHODS = ['beam20', 'sampling_verifier20', 'self_refine_r1', 'tot_lm_vote',
           'got', 's1', 'ev', 'fewshot', 'reflexion']
IND_TASKS = ['BDP', 'BDQ', 'BPQ', 'DPQ', 'BDPQ']
OOD_TASKS = ['MPQ', 'BDMQ', 'BHMQ', 'BMPQ', 'HMPQ']
EXPECTED_INPUTS = 3905


def validate_output_properties(candidate, source, context):
    """An unknown property loss is an infrastructure error, not a failed answer."""
    assert type(candidate['rdkit_valid']) is bool, context
    assert isinstance(candidate['raw_smiles'], str), context
    if (candidate['rdkit_valid'] and candidate['raw_smiles'].strip()
            and not KERNEL['finite_values'](candidate.get('properties'), list(source))):
        reason = known_failure(candidate)
        if reason is None or candidate.get('properties') is not None:
            raise ValueError('Unknown/partial target-property absence: ' + repr(context))
        return reason
    return None


def archive_stale_tables():
    """Keep previous artifacts recoverable, but never present them as current."""
    paths = [ROOT / 'results' / ('table1.' + ext) for ext in ('json', 'csv', 'tex')]
    paths = [p for p in paths if p.exists()]
    if not paths:
        return []
    archive = ROOT / 'results/stale_table1' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    archive.mkdir(parents=True, exist_ok=False)
    for path in paths:
        path.replace(archive / path.name)
    return [str(archive / p.name) for p in paths]


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    cfg = json.loads((ROOT / 'config/methods.json').read_text())
    assert cfg['models'] == MODELS, 'Fixed two-model identity/order changed'
    assert cfg['methods'] == METHODS, 'Fixed nine-method identity/order changed'
    assert cfg['tasks'] == IND_TASKS + OOD_TASKS, 'Fixed IND/OOD task identity/order changed'
    manifest_spec = json.loads((ROOT / 'config/manifests.json').read_text())
    assert set(manifest_spec) == set(cfg['tasks']), 'Missing/extra task manifest'
    manifests = {}
    for task, spec in manifest_spec.items():
        data = Path(spec['path']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == spec['sha256'], task
        records = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
        manifests[task] = {r['input_uid']: r for r in records}
        assert len(manifests[task]) == len(records) == spec['inputs']
        assert [r['input_idx'] for r in records] == list(range(len(records))), task
        assert all(r['task_id'] == task for r in records), task
    assert sum(map(len, manifests.values())) == EXPECTED_INPUTS
    groups = defaultdict(dict)
    source_files = {}
    accepted_property_failures = []
    for path in sorted((ROOT / 'results').glob('*.jsonl.gz')):
        source_files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with gzip.open(str(path), 'rt') as handle:
            for line in handle:
                row = json.loads(line)
                assert row['schema_version'] == cfg['schema'], str(path)
                model, method, task, uid = (row[k] for k in ('model_id', 'method_id', 'task_id', 'input_uid'))
                assert model in cfg['models'] and method in cfg['methods'] and task in cfg['tasks']
                record = manifests[task][uid]
                assert record['input_idx'] == row['input_idx']
                assert record['source_properties'] == row['source_properties']
                assert uid not in groups[model, method, task], (model, method, task, uid)
                candidate = row['selected_candidate']
                assert candidate['original_slot_id'] == 0
                assert row.get('selection_provenance'), (model, method, task, uid)
                for key in ('input_uid', 'input_idx', 'source_smiles'):
                    if key in candidate:
                        assert candidate[key] == record[key], (model, method, task, uid, key)
                if 'model_id' in candidate:
                    assert candidate['model_id'] == model
                reason = validate_output_properties(candidate, row['source_properties'], (model, method, task, uid))
                if reason is not None:
                    accepted_property_failures.append(dict(model_id=model, method_id=method,
                                                           task_id=task, input_uid=uid, reason=reason))
                computed = evaluate_registered(task, uid, row['source_properties'], [candidate], 1)
                supplied = row['evaluation']
                for key in ('SR_dir_success', 'SR_delta_success', 'n_slots', 'selected_candidate_id_dir',
                            'selected_smiles_dir', 'RI_legacy', 'RI_abs_eps01'):
                    assert computed[key] == supplied[key], (model, method, task, uid, key)
                assert supplied['n_slots'] == 1 and supplied['candidates'] == [candidate]
                groups[model, method, task][uid] = computed
    progress, task_metrics, complete_configs, comparison = [], [], [], []
    for method in cfg['methods']:
        paired = {'method_id': method}
        for model in cfg['models']:
            finished = {}
            n = 0
            for task in cfg['tasks']:
                actual = groups[model, method, task]
                expected = manifests[task]
                assert set(actual).issubset(expected)
                n += len(actual)
                complete = set(actual) == set(expected)
                progress.append(dict(model_id=model, method_id=method, task_id=task,
                                     completed=len(actual), expected=len(expected), complete=complete))
                if complete:
                    rows = [actual[uid] for uid in expected]
                    metrics = aggregate_registered(rows)
                    task_metrics.append(dict(model_id=model, method_id=method, task_id=task, metrics=metrics))
                    finished[task] = 100.0 * sum(r['SR_dir_success'] for r in rows) / len(rows)
            paired[model + '_completed_inputs'] = n
            if len(finished) == 10:
                complete_configs.append((model, method))
                paired[model + '_IND'] = sum(finished[t] for t in IND_TASKS) / 5
                paired[model + '_OOD'] = sum(finished[t] for t in OOD_TASKS) / 5
                paired[model + '_Macro10'] = sum(finished.values()) / 10
        comparison.append(paired)
    complete = len(complete_configs) == 18
    audit = dict(status='COMPLETE_FULL3905_TWO_MODELS_NINE_METHODS_K1' if complete else 'INCOMPLETE_NOT_A_FINAL_MAIN_TABLE',
                 checked_utc=datetime.now(timezone.utc).isoformat(),
                 complete_model_method_configurations=len(complete_configs), expected_configurations=18,
                 final_records=sum(p['completed'] for p in progress), expected_records=18 * EXPECTED_INPUTS,
                 task_groups=progress, source_files=source_files,
                 unknown_property_failures=0, accepted_property_failures=accepted_property_failures,
                 scope='Same final output count and evaluation; method-specific internal compute and public property-tool use differ.')
    if not complete:
        audit['archived_stale_table_artifacts'] = archive_stale_tables()
    save(ROOT / 'results/endpoint_audit.json', audit)
    save(ROOT / 'results/task_metrics.json', task_metrics)
    if not complete:
        save(ROOT / 'results/incomplete_progress.json', comparison)
        print(json.dumps({k: audit[k] for k in ('status', 'complete_model_method_configurations', 'final_records')}))
        if not args.partial:
            raise SystemExit('Refusing final Table1: some required endpoints are missing.')
        return
    assert audit['final_records'] == 18 * EXPECTED_INPUTS
    save(ROOT / 'results/table1.json', comparison)
    fields = ['method_id'] + [m + '_' + key for m in cfg['models'] for key in ('IND', 'OOD', 'Macro10')]
    with (ROOT / 'results/table1.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(comparison)
    names = {'beam20':'Beam (width20)', 'sampling_verifier20':'Sampling + task verifier',
             'self_refine_r1':'Self-Refine (r1)', 'tot_lm_vote':'ToT (LM voting)',
             'got':'GoT (G12)', 's1':'s1-style budget forcing', 'ev':'Exact Voting (N400)',
             'fewshot':'Few-shot ICL', 'reflexion':'Reflexion (max3)'}
    latex = ['% Complete 3905-input, two-model, nine-method K1 result; generated from audited endpoints.',
             r'\noindent\begin{minipage}{\linewidth}', r'\centering',
             r'\begin{tabular}{lrrrrrr}', r'\toprule',
             r'& \multicolumn{3}{c}{Mistral} & \multicolumn{3}{c}{Llama} \\',
             r'Method & IND & OOD & Macro10 & IND & OOD & Macro10 \\', r'\midrule']
    for row in comparison:
        values = [row[m+'_'+key] for m in cfg['models'] for key in ('IND','OOD','Macro10')]
        latex.append(names[row['method_id']] + ' & ' + ' & '.join('%.2f' % v for v in values) + r' \\')
    latex += [r'\bottomrule', r'\end{tabular}', r'\par\smallskip',
              r'\raggedright\footnotesize',
              r'\textit{Protocol.} Each model--method pair uses all 3,905 inputs and returns one output ($K=1$); failed outputs remain in the denominator. '
              r'IND and OOD average five tasks each; Macro10 equally averages all ten tasks. Internal compute budgets differ.',
              r'\par Public property tools are available during inference. Sampling uses them for terminal verification, '
              r'GoT for graph search and final ranking, and Reflexion for feedback and stopping. '
              r'Sampling examines the fixed first 20 of 40 generated proposals; its verifier uses the final proxy-property success criterion, '
              r'so its $K=1$ success rate equals prefix-20 coverage. This is not independent experimental property validation.',
              r'\par All rows are molecular-task adaptations. Self-Refine uses one feedback--revision cycle; '
              r'ToT uses model voting; GoT uses the declared task scorer. '
              r's1 denotes the budget-forcing component port without s1K training; few-shot ICL is a baseline.',
              r'\end{minipage}']
    (ROOT / 'results/table1.tex').write_text('\n'.join(latex)+'\n')
    print(json.dumps({k: audit[k] for k in ('status', 'complete_model_method_configurations', 'final_records')}))


if __name__ == '__main__':
    main()
