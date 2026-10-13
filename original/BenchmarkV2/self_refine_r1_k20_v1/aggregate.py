"""Publish K20 metrics only with exactly twenty r1 terminals for every manifest input."""
import csv
import copy
from common import *
from metrics_entry import evaluate_registered, aggregate_registered


def aggregate_pool(record, candidates):
    assert len(candidates) == 20
    assert [c['rep_seed'] for c in candidates] == list(range(51001, 51021))
    normalized = []
    for slot, c in enumerate(candidates):
        assert c['original_slot_id'] == 1 and c['task_id'] == record['task_id']
        assert c['input_uid'] == record['input_uid'] and c['source_smiles'] == record['source_smiles']
        d = copy.deepcopy(c)
        d.update(source_candidate_id=c['candidate_id'], source_method_id=c['method_id'], generated_round=1,
                 original_slot_id=slot, visible_slot_id=slot, method_id=METHOD)
        d['candidate_id'] = 'sr-r1-k20-visible-' + digest([c['candidate_id'], slot, METHOD])
        normalized.append(d)
    result = evaluate_registered(record['task_id'], record['input_uid'], record['source_properties'], normalized, 20)
    result['interface_version'] = 'SelfRefine-r1-terminal-K20-v1'
    return result


def main():
    require_allocation()
    verify()
    assert read(ROOT / 'probe_gate.json')['status'] == 'PASS'
    task_metrics = {}
    total_slots = 0
    for task in TASKS:
        byuid = {r['input_uid']: [] for r in manifest(task)}
        native_success = {r['input_uid']: [] for r in manifest(task)}
        for seed in range(51001, 51021):
            if seed <= 51003:
                data = read(ROOT / 'reuse' / str(seed) / (task + '.json.gz'))
            else:
                assert read(ROOT / 'evaluation' / str(seed) / 'complete.json')['status'] == 'complete'
                data = read(ROOT / 'evaluation' / str(seed) / task / 'result.json.gz')
            assert len(data['rows']) == len(byuid)
            assert {r['input_uid'] for r in data['rows']} == set(byuid)
            for row in data['rows']:
                assert row['r'] == 1 and row['rep_seed'] == seed
                byuid[row['input_uid']].append(row['candidate'])
                native_success[row['input_uid']].append(row['metric_row']['SR_dir_success'])
        metrics = []
        for record in manifest(task):
            row = aggregate_pool(record, byuid[record['input_uid']])
            assert row['SR_dir_success'] == any(native_success[record['input_uid']])
            metrics.append(row)
        task_metrics[task] = aggregate_registered(metrics)
        total_slots += task_metrics[task]['n_slots']
        destination = ROOT / 'results' / (task + '_K20.json.gz')
        if not destination.exists():
            save(destination, dict(method_id=METHOD, task_id=task, status='complete', k=20, refinements=1,
                 metrics=task_metrics[task], rows=metrics, plan_sha256=sha(ROOT / 'plan.json')))
        print(task, task_metrics[task]['SR_dir'], flush=True)
    assert total_slots == 78100
    costs = {str(seed): read(ROOT / 'generation' / str(seed) / 'complete.json') for seed in range(51004, 51021)}
    result = dict(status='COMPLETE_ALL3905_K20', method_id=METHOD, label='Self-Refine (r=1, K=20)',
                  task_inputs=3905, final_output_slots=total_slots, task_metrics=task_metrics,
                  Macro10_SR_dir=sum(m['SR_dir'] for m in task_metrics.values())/10,
                  Macro10_SR_delta=sum(m['SR_delta'] for m in task_metrics.values())/10,
                  reused_trajectories=11715, new_trajectories=66385, new_generation_costs=costs,
                  comparison='same model, inputs, K20 slots and success criteria; unequal internal computation',
                  limitations='r1 budget adaptation; serial legacy prefixes plus native HF batched new repeats; no r4 K20 claim')
    save(ROOT / 'results' / 'summary.json', result)
    with (ROOT / 'results' / 'table1_self_refine_row.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Method', 'r', 'K'] + TASKS + ['Macro10'])
        writer.writeheader()
        writer.writerow(dict(Method='Self-Refine', r=1, K=20, Macro10=result['Macro10_SR_dir'],
                             **{t: task_metrics[t]['SR_dir'] for t in TASKS}))
    print(json.dumps({k:v for k,v in result.items() if k not in ('task_metrics', 'new_generation_costs')}), flush=True)


if __name__ == '__main__':
    main()
