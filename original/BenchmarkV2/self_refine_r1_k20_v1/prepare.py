"""Freeze scope and execution order, and reuse all three saved r1 endpoints."""
import ast
import copy
import json
import time
from pathlib import Path
from common import *
from metrics_entry import evaluate_registered, aggregate_registered
from trajectory_evidence import audit_trajectory
from self_refine import candidate as old_candidate


def main():
    assert not (ROOT / 'plan.json').exists(), 'Already frozen'
    cfg = read(LEGACY / 'configs/science.json')
    science = {k: cfg[k] for k in ('decoding', 'max_new_tokens', 'tokenizer_fast', 'padding_side',
                                  'context_limit', 'metric_version', 'delta_version', 'prompt_adaptation')}
    science.update(method_id=METHOD, execution_id=EXECUTION, refinements=1, visible_k=20,
                   initial_in_refinements=False, model_calls_per_trajectory=4, inference_tool_calls=0,
                   stop='fixed after feedback1; no success-based stop', output='one y1 per trajectory',
                   reuse_seeds=[51001, 51002, 51003], new_seeds=list(range(51004, 51021)),
                   output_slot_order=list(range(51001, 51021)))
    cfg_plan = dict(science=science, scientific_identity_sha256=digest(science),
                    runtime=read(LEGACY / 'configs/runtime.json'), total_inputs=3905,
                    new_trajectories=66385, batch_size=80, generation_array_count=17, max_gpus=2,
                    scheduling='one full manifest per seed; array concurrency 2; CPU evaluation concurrency 4',
                    batch_order='original task order and input_idx, across task boundaries; last batch 65',
                    rng='SHA256 SR-R1-K20-BATCH-v1/batch_id/ordered_UID_digest/role/r; fixed native HF sampling',
                    resume='reuse complete calls/batches after identity verification; restart unfinished call with same batch and seed',
                    batch_equivalence='same model, prompts, stochastic decoding and parse rules; batched numerics/RNG differ from serial legacy; no bitwise equivalence claim',
                    old_reuse='all three saved r1 prefixes, no selection by quality; old r4 outputs excluded',
                    quality_retries=False, invalid_outputs='retain as failed slots; never resample',
                    evaluation='original property producers and registered guarded metric; CPU only after generation',
                    scope='Self-Refine only; no full runs of other methods; no manuscript numbers before full coverage',
                    authorization='User accepted r1 K20 and explicitly requested audit, then launch if sound',
                    timing='Prior timing estimates are planning values, not completed production measurements',
                    created_unix=time.time())
    assert len(records()) == 3905
    execution = []
    seeds = []
    for seed in science['new_seeds']:
        for batch in batches(seed):
            entry = dict(batch)
            entry['rows'] = [(r['task_id'], r['input_uid']) for r in batch['rows']]
            entry['call_seeds'] = [{'role': role, 'r': r, 'seed': batch_seed(batch, role, r)} for role, r in ROLES]
            seeds.extend(c['seed'] for c in entry['call_seeds'])
            execution.append(entry)
    assert len(seeds) == len(set(seeds)), 'Batch RNG collision'
    assert sum(len(b['rows']) for b in execution) == 66385
    save(ROOT / 'batch_manifest.json', execution)
    cfg_plan['batch_manifest_sha256'] = sha(ROOT / 'batch_manifest.json')
    save(ROOT / 'plan.json', cfg_plan)

    groups = read(LEGACY / 'results/budget_150_groups.json')
    sources = []
    audit = dict(status='PASS', inputs=3905, reused_candidates=0, trajectory_spot_checks=0,
                 basis='full r1 coverage, candidate identity and metrics checked; first/last trajectory per task-seed checked against stored generation hashes')
    pooled = {task: {} for task in TASKS}
    for seed in science['reuse_seeds']:
        for task in TASKS:
            group, = [g for g in groups if g['task_id'] == task and g['rep_seed'] == seed and g['r'] == 1]
            assert group['status'] == 'complete'
            path = Path(group['result_path'])
            sources.append(dict(path=str(path), sha256=sha(path)))
            data = read(path)
            rows = [row for row in data['rows'] if row['r'] == 1]
            recs = manifest(task)
            assert len(rows) == group['expected_n'] == group['completed_n'] == len(recs)
            assert [r['input_uid'] for r in rows] == [r['input_uid'] for r in recs]
            assert all(row['prefix_generation_cost']['model_calls'] == 4 for row in rows)
            for row, record in zip(rows, recs):
                c = row['candidate']
                assert c['original_slot_id'] == 1 and c['rep_seed'] == seed and c['task_id'] == task
                assert c['source_smiles'] == record['source_smiles']
                assert c['parse'] == parse_answer(c['raw_text']) and c['raw_smiles'] == c['parse']['raw_smiles']
                native = dict(c, original_slot_id=0, visible_slot_id=0)
                metric = evaluate_registered(task, record['input_uid'], record['source_properties'], [native], 1)
                assert metric['SR_dir_success'] == row['metric_row']['SR_dir_success']
                pooled[task].setdefault(record['input_uid'], []).append(c)
            assert sum(row['metric_row']['SR_dir_success'] for row in rows) == group['metrics']['n_success_dir']
            for index in (0, len(recs) - 1):
                source = data['trajectory_source_index'][index]
                assert sha(source['path']) == source['sha256']
                trajectory = audit_trajectory(source['path'], recs[index], seed, deep=True)
                expected = old_candidate(recs[index], seed, 1, trajectory['answers'][1])
                assert expected['candidate_id'] == rows[index]['candidate_id']
                audit['trajectory_spot_checks'] += 1
            save(ROOT / 'reuse' / str(seed) / (task + '.json.gz'), dict(task_id=task, rep_seed=seed,
                 source_result=str(path), source_sha256=sources[-1]['sha256'], rows=rows))
            audit['reused_candidates'] += len(rows)
            print('Verified existing r1', task, seed, len(rows), flush=True)
    assert audit['reused_candidates'] == 11715
    macro = []
    for task in TASKS:
        metrics = []
        for rec in manifest(task):
            pool = [dict(c, original_slot_id=i, visible_slot_id=i) for i, c in enumerate(pooled[task][rec['input_uid']])]
            out = evaluate_registered(task, rec['input_uid'], rec['source_properties'], pool, 3)
            assert out['SR_dir_success'] == any(evaluate_registered(task, rec['input_uid'], rec['source_properties'],
                     [dict(c, original_slot_id=0)], 1)['SR_dir_success'] for c in pool)
            metrics.append(out)
        macro.append(aggregate_registered(metrics)['SR_dir'])
    audit['K3_OR_consistency_macro10'] = sum(macro) / len(macro)
    assert abs(audit['K3_OR_consistency_macro10'] - 59.660214748424885) < 1e-10
    save(ROOT / 'reuse_audit.json', audit)
    save(ROOT / 'legacy_sources.json', sources)

    # Freeze code/configs and protected scientific dependencies, not historical approval gates.
    inventory_paths = set(ROOT.glob('*.py')) | {ROOT / 'plan.json', ROOT / 'batch_manifest.json'}
    inventory_paths |= set((ROOT / 'reuse').glob('*/*.json.gz'))
    inventory_paths |= set((LEGACY / 'src').glob('*.py'))
    inventory_paths |= {LEGACY / n for n in ('admet_worker.py', 'eval_worker.py', 'configs/prompts.json',
                        'configs/science.json', 'configs/properties.json', 'configs/runtime.json',
                        'evidence/read_set.json', 'evidence/scientific_identity.json')}
    inventory_paths |= {Path(e['path']) for e in read(LEGACY / 'evidence/read_set.json')}
    for task in TASKS:
        inventory_paths.add(Path(task_contract(task)['manifest_path']))
    for name in ('model_metadata_before.json', 'property_metadata_before.json'):
        inventory_paths.add(LEGACY / 'evidence' / name)
        for e in read(LEGACY / 'evidence' / name):
            st = Path(e['path']).stat()
            assert st.st_size == e['size'] and st.st_mtime_ns == e['mtime_ns'], e['path']
    save(ROOT / 'inventory.json', [dict(path=str(p), sha256=sha(p)) for p in sorted(inventory_paths)])
    verify()
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
