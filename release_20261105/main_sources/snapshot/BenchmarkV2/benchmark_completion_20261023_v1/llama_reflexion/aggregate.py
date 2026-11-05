"""Audit complete terminal identities and derive fixed-slot K1 and full K20."""
from common import *
from metrics_entry import evaluate_registered, aggregate_registered
sys.path.insert(0, str(ROOT.parent / 'recovery_20261027_v1'))
from property_repair import terminal_candidate


def main():
    require_allocation()
    verify()
    all_rows, tasks, candidate_ids = [], {}, set()
    for task in TASKS:
        rows20, rows1 = [], []
        for unit in units():
            if unit['task_id'] != task:
                continue
            directory = ROOT/'runs'/unit['unit_id']
            receipt = read(directory/'complete.json')
            result = read(directory/'result.json.gz')
            assert receipt['status'] == 'complete' and result['unit'] == unit
            assert receipt['plan_sha256'] == result['plan_sha256'] == sha(ROOT/'plan.json')
            assert len(result['rows']) == len(records(unit))
            for row, record in zip(result['rows'], records(unit)):
                assert row['input_uid'] == record['input_uid']
                pool = row['candidates']
                assert [c['original_slot_id'] for c in pool] == list(range(20))
                for c in pool:
                    assert c['model_id'] == 'gellmo_p6_llama' and c['candidate_id'] not in candidate_ids
                    assert c['source_smiles'] == record['source_smiles'] and c['input_uid'] == record['input_uid']
                    candidate_ids.add(c['candidate_id'])
                assert row == evaluate_registered(task, record['input_uid'], record['source_properties'], pool, 20)
                rows20.append(row)
                rows1.append(evaluate_registered(task, record['input_uid'], record['source_properties'], pool[:1], 1))
                final_pool = []
                for c in pool:
                    final_pool.append(terminal_candidate(c, record['source_properties']))
                all_rows.append(dict(model_id='gellmo_p6_llama', method_id='reflexion-max3-sampled-R20',
                    task_id=task, input_uid=record['input_uid'], input_idx=record['input_idx'],
                    source_smiles=record['source_smiles'], source_properties=record['source_properties'], candidates=final_pool))
        assert [r['input_uid'] for r in rows20] == [r['input_uid'] for r in manifest(task)]
        tasks[task] = dict(K1=aggregate_registered(rows1), K20=aggregate_registered(rows20))
        save(ROOT/'results'/(task+'.json.gz'), dict(K1=rows1, K20=rows20, metrics=tasks[task]))
    assert len(all_rows) == 3905 and len(candidate_ids) == 78100
    save(ROOT/'results/terminal_inputs.json.gz', all_rows)
    output = ROOT/'results/terminal_inputs.jsonl.gz'
    if output.exists():
        with gzip.open(str(output), 'rt') as stream:
            assert [json.loads(line) for line in stream] == all_rows
    else:
        tmp = output.with_name('.terminal_inputs.'+str(os.getpid())+'.tmp')
        with tmp.open('xb') as stream:
            with gzip.GzipFile(fileobj=stream, mode='wb', mtime=0) as compressed:
                for record in all_rows:
                    compressed.write((json.dumps(record, allow_nan=False, separators=(',', ':'))+'\n').encode())
        try:
            os.link(str(tmp), str(output))
        finally:
            tmp.unlink()
    verify()
    save(ROOT/'reports/final_audit.json', dict(status='PASS_COMPLETE_TERMINAL_PROPERTIES',
        model_id='gellmo_p6_llama', method_id='reflexion-max3-sampled-R20', inputs=3905,
        terminal_slots=78100, unknown_property_failures=0, source_files_unchanged=True,
        output_path=str(output), output_sha256=sha(output),
        final_trial_properties_reused=True, new_terminal_property_predictions=0,
        known_molecular_failure_slots=sum(c['known_molecular_property_failure'] for r in all_rows for c in r['candidates']),
        plan_sha256=sha(ROOT/'plan.json'), freeze_sha256=sha(ROOT/'freeze_v4.json')))
    save(ROOT/'results/summary.json', dict(status='COMPLETE_ALL3905_K1_K20', inputs=3905,
        terminal_slots=78100, model_id='gellmo_p6_llama', method_id='reflexion-max3-sampled-R20',
        task_metrics=tasks, Macro10={k:sum(v[k]['SR_dir'] for v in tasks.values())/10 for k in ('K1','K20')},
        plan_sha256=sha(ROOT/'plan.json'), terminal_inputs_sha256=sha(ROOT/'results/terminal_inputs.json.gz')))
    print(json.dumps(dict(status='COMPLETE_ALL3905_K1_K20', inputs=3905, terminal_slots=78100)))


if __name__ == '__main__':
    main()
