"""Read existing r1 prefixes; never generate, alter, or select between trajectories."""
from self_refine_common import *
from contracts import manifest
from self_refine import candidate, call_seed, prefix_cost, prompt_for


def main():
    groups = read(MISTRAL / 'results/budget_150_groups.json')
    dirs = sorted(p for p in (LLAMA / 'runs').iterdir()
                  if p.is_dir() and p.name.startswith('llama_self_refine_'))
    templates = read(LLAMA / 'configs/p5_prompts.json')
    audit = dict(status='PASS', seed=51001, round=1, visible_k=1, models={})
    for task in TASKS:
        records = manifest(task)
        group, = [g for g in groups if g['task_id'] == task and g['rep_seed'] == 51001 and g['r'] == 1]
        assert group['status'] == 'complete'
        source = Path(group['result_path'])
        source_sha = sha(source)
        original = read(source)
        rows = [row for row in original['rows'] if row['r'] == 1]
        assert [r['input_uid'] for r in rows] == [r['input_uid'] for r in records]
        endpoints = []
        for record, row in zip(records, rows):
            c = row['candidate']
            assert c['original_slot_id'] == 1 and c['rep_seed'] == 51001
            assert row['prefix_generation_cost']['model_calls'] == 4
            result = endpoint(record, c, 'gellmo_p6_mistral', dict(
                rule='fixed_seed51001_refine1_prefix_including_feedback1_no_cross_round_selection',
                source_path=str(source), source_sha256=source_sha, original_round=1,
                prefix_generation_cost=row['prefix_generation_cost']))
            assert result['evaluation']['SR_dir_success'] == row['metric_row']['SR_dir_success']
            endpoints.append(result)
        write(ROOT / 'results' / ('self_refine_mistral__' + task + '.json.gz'), endpoints)
        audit['models'].setdefault('mistral', {})[task] = len(endpoints)

        indexed = {}
        for directory in dirs:
            if not directory.name.startswith('llama_self_refine_' + task + '_'):
                continue
            for record in read(directory / 'inputs.json'):
                assert record['task_id'] == task
                folder = directory / 'trajectories' / ('{:04d}'.format(record['input_idx']))
                full = read(folder / 'complete.json')
                answer = read(folder / 'refine1.json')
                assert full['record'] == record and full['rep_seed'] == 51001 and full['complete']
                assert full['answers'][1] == answer
                assert answer['role'] == 'refine' and answer['r'] == 1
                assert answer['seed'] == call_seed(51001, task, record['input_uid'], 'refine', 1)
                # Verify the complete prefix including the final feedback required by original stop.
                answers, feedbacks = [], []
                for role, r in [('init', 0), ('feedback', 0), ('refine', 1), ('feedback', 1)]:
                    value = read(folder / (role + str(r) + '.json'))
                    assert value['prompt_digest'] == digest(prompt_for(role, full['task_text'], answers, feedbacks, r, templates))
                    assert value == (full['feedbacks'][r] if role == 'feedback' else full['answers'][r])
                    (feedbacks if role == 'feedback' else answers).append(value)
                c = candidate(record, 51001, 1, answer, full['scientific_identity_sha256'])
                provenance = dict(rule='fixed_seed51001_refine1_prefix_including_feedback1_no_cross_round_selection',
                                  source_path=str(folder / 'refine1.json'), source_sha256=sha(folder / 'refine1.json'),
                                  trajectory_path=str(folder / 'complete.json'), trajectory_sha256=sha(folder / 'complete.json'),
                                  original_round=1, prefix_generation_cost=prefix_cost(full, 1))
                assert record['input_uid'] not in indexed
                indexed[record['input_uid']] = dict(record=record, candidate=c, provenance=provenance)
        assert set(indexed) == {r['input_uid'] for r in records}
        payload = [indexed[r['input_uid']] for r in records]
        assert [x['record'] for x in payload] == records
        # Exact raw/source/contract identities permit reuse across r4/r1 without quality selection.
        cache_sources = []
        for directory in dirs:
            if directory.name.startswith('llama_self_refine_' + task + '_'):
                cache_sources.extend(directory.glob('properties/event*_batch*/standard/responses_raw.json'))
        write(ROOT / 'runs' / 'self_refine_llama' / task / 'prepared.json.gz', dict(
            task_id=task, records=records, candidates=[x['candidate'] for x in payload],
            provenance=[x['provenance'] for x in payload],
            property_cache_sources=[dict(path=str(p), sha256=sha(p)) for p in sorted(cache_sources)]))
        audit['models'].setdefault('llama', {})[task] = len(payload)
        print(task, 'Mistral exported; Llama r1 prepared', len(payload), flush=True)
    assert all(sum(counts.values()) == 3905 for counts in audit['models'].values())
    write(ROOT / 'reports/self_refine_prefix_preparation.json', audit)


if __name__ == '__main__':
    main()
