"""Publish a nine-method table only after the three full K20 pools are complete."""
import csv
from common import *
from metrics_entry import evaluate_registered, aggregate_registered


def main():
    require_allocation()
    verify()
    totals = {}
    table = []
    old_root = REPO / 'BenchmarkV2/advisory_deliverables/core_tables_revision_20260918_v1'
    old = read(old_root / 'table_data.json')
    keep = ['beam20', 'p2_bon40_prefix20', 'p6_tot20', 'p8_g12_k20']
    compact = {m: [] for m in keep}
    with gzip.open(str(old_root / 'main_per_input_compact.jsonl.gz'), 'rt') as f:
        for line in f:
            row = json.loads(line)
            if row['method_id'] in compact:
                compact[row['method_id']].append(row)
    for m in keep:
        source = next(x for x in old['table1'] if x['method_id'] == m)
        assert source['K'] == 20 and len(compact[m]) == 3905
        row = dict(Method=source['method'], Configuration=m, K=20, **{t: source[t] for t in TASKS})
        for task in TASKS:
            part = [r for r in compact[m] if r['task_id'] == task]
            assert {r['input_uid'] for r in part} == {r['input_uid'] for r in manifest(task)}
            assert len(part) == len(manifest(task))
        table.append(row)
    p5 = read(REPO / 'BenchmarkV2/self_refine_r1_k20_v1/results/summary.json')
    assert p5['status'] == 'COMPLETE_ALL3905_K20' and p5['final_output_slots'] == 78100
    table.append(dict(Method='Self-Refine', Configuration='r1;20 terminal trajectories', K=20,
                      **{t:p5['task_metrics'][t]['SR_dir'] for t in TASKS}))
    p10 = read(REPO / 'BenchmarkV2/k20_nine_full_20261003_v1/p10_comparison/final_comparison.json')
    assert p10['status'] == 'EXACT_FULL_3905_PAIRED_COMPARISON'
    p10row = dict(Method='Exact voting', Configuration='20 votes x20 consecutive disjoint groups', K=20,
                  **{t['task']:t['new20x20_K20_SR'] for t in p10['tasks']})
    labels = dict(fewshot='Few-shot ICL', s1='s1 Budget Forcing', reflexion='Reflexion')
    for method in ['s1', 'fewshot', 'reflexion']:
        tasks = {}
        nslots = 0
        for task in TASKS:
            rows = []
            for unit in units():
                if unit['method'] != method or unit['task_id'] != task:
                    continue
                directory = ROOT / 'runs' / unit['unit_id']
                assert read(directory / 'complete.json')['status'] == 'complete'
                data = read(directory / 'result.json.gz')
                assert data['unit'] == unit and data['plan_sha256'] == sha(ROOT / 'plan.json')
                for row, record in zip(data['rows'], records(unit)):
                    assert row['input_uid'] == record['input_uid']
                    pool = row['candidates']
                    assert len(pool) == 20 and [c['original_slot_id'] for c in pool] == list(range(20))
                    assert all(c['input_uid'] == record['input_uid'] and c['source_smiles'] == record['source_smiles'] for c in pool)
                    recomputed = evaluate_registered(task, record['input_uid'], record['source_properties'], pool, 20)
                    assert recomputed == row
                    rows.append(row)
            assert [r['input_uid'] for r in rows] == [r['input_uid'] for r in manifest(task)]
            metrics = aggregate_registered(rows)
            tasks[task] = metrics
            nslots += metrics['n_slots']
            save(ROOT / 'results' / (method + '_' + task + '.json.gz'),
                 dict(status='complete', method=method, task=task, k=20, rows=rows, metrics=metrics))
        assert nslots == 78100
        totals[method] = dict(inputs=3905, slots=nslots, task_metrics=tasks,
                              Macro10_SR_dir=sum(t['SR_dir'] for t in tasks.values()) / 10)
        table.append(dict(Method=labels[method], Configuration=plan()['extensions'][method]['main_id'], K=20,
                          **{t:tasks[t]['SR_dir'] for t in TASKS}))
        if method == 's1':
            table.append(p10row)
    assert len(table) == 9
    for row in table:
        row['Macro10'] = sum(row[t] for t in TASKS) / 10
    result = dict(status='COMPLETE_NINE_METHODS_FULL3905_K20', table1=table, new_methods=totals,
                  plan_sha256=sha(ROOT / 'plan.json'), same_inputs=3905, evaluated_k=20,
                  primary_metric='any submitted terminal strictly improves every requested target after original round2',
                  comparison='matched output count/model/inputs/success rule; unequal compute and feedback access disclosed',
                  legacy_note='Preserve documented four adjudicated ToT evaluation failures',
                  tables2_and3='unchanged historical configurations; their old costs must not be attributed to the new K20 rows',
                  manuscript_updated=False)
    save(ROOT / 'results' / 'summary.json', result)
    with (ROOT / 'results' / 'table1_nine_methods_k20.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Method', 'Configuration', 'K'] + TASKS + ['Macro10'])
        writer.writeheader()
        writer.writerows(table)
    print(json.dumps(dict(status=result['status'], table=table)), flush=True)


if __name__ == '__main__':
    main()
