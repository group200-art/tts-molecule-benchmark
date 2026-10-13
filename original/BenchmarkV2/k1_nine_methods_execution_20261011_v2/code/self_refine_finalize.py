"""Publish full-coverage endpoint exports only when all ten tasks are complete."""
import argparse
from self_refine_common import *
from cache_metrics import aggregate_registered
from contracts import manifest


def finalize(model):
    rows, groups = [], []
    for task in TASKS:
        source = ROOT / 'results' / ('self_refine_' + model + '__' + task + '.json.gz')
        part = read(source)
        records = manifest(task)
        assert [r['input_uid'] for r in part] == [r['input_uid'] for r in records]
        assert all(r['selected_candidate']['rep_seed'] == 51001 and
                   r['selected_candidate']['source_original_slot_id'] == 1 and
                   r['evaluation']['n_slots'] == 1 for r in part)
        groups.append(dict(task_id=task, model_id=part[0]['model_id'], method_id='self_refine_r1',
                           n_inputs=len(part), source_path=str(source), source_sha256=sha(source),
                           metrics=aggregate_registered([r['evaluation'] for r in part])))
        rows.extend(part)
    assert len(rows) == 3905 and len({(r['task_id'], r['input_uid']) for r in rows}) == 3905
    out = check_output(ROOT / 'results' / ('self_refine_' + model + '__self_refine_r1.jsonl.gz'))
    blob = ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows).encode()
    if out.exists():
        with gzip.open(str(out), 'rb') as handle:
            assert handle.read() == blob
    else:
        with out.open('xb') as handle:
            with gzip.GzipFile(fileobj=handle, mode='wb', mtime=0) as compressed:
                compressed.write(blob)
    write(ROOT / 'results' / ('self_refine_' + model + '__group_metrics.json'), groups)
    receipt = dict(status='PASS_FULL3905_K1_R1_SEED51001', model_id=rows[0]['model_id'],
                   n_inputs=3905, n_tasks=10, output_path=str(out), output_sha256=sha(out),
                   macro10_SR_dir=sum(g['metrics']['SR_dir'] for g in groups) / 10,
                   IND_SR_dir=sum(g['metrics']['SR_dir'] for g in groups[:5]) / 5,
                   OOD_SR_dir=sum(g['metrics']['SR_dir'] for g in groups[5:]) / 5)
    write(ROOT / 'reports' / ('self_refine_' + model + '_complete.json'), receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['mistral', 'llama', 'both'], default='both')
    args = parser.parse_args()
    for model in (['mistral', 'llama'] if args.model == 'both' else [args.model]):
        finalize(model)
