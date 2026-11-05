"""Fill four missing workload rows from committed ToT/Reflexion trajectories."""
import csv
import gzip
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BENCH = ROOT.parent
RUN = BENCH / 'benchmark_completion_20261023_v1'
DATA = ROOT / 'paper/data'
TASKS = ['BDP', 'BDQ', 'BPQ', 'DPQ', 'BDPQ', 'MPQ', 'BDMQ', 'BHMQ', 'BMPQ', 'HMPQ']
SOURCES = {}
PER_INPUT = []


def read(path, expected_sha=None):
    path = Path(path).resolve()
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha is not None:
        assert digest == expected_sha, str(path)
    if str(path) in SOURCES:
        assert SOURCES[str(path)] == digest
    SOURCES[str(path)] = digest
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def token_count(response):
    count = response['output_tokens']
    assert type(count) is int and count > 0
    if 'generated_ids' in response:
        assert len(response['generated_ids']) == count
    return count


def reflexion(model, records):
    directory = RUN / (model.lower() + '_reflexion')
    units = [u for u in read(directory / 'units.json') if u['method'] == 'reflexion']
    assert len(units) == 56
    seen, identities, totals, byinput = set(), set(), Counter(), {}
    for unit in units:
        base = directory / 'runs' / unit['unit_id']
        complete = read(base / 'complete.json')
        assert complete['status'] == 'complete' and complete['inputs'] == unit['stop'] - unit['start']
        for b in range((unit['stop'] - unit['start'] + 3) // 4):
            batch = read(base / (unit['unit_id'] + '_b%02d' % b) / 'generation.json.gz')
            assert batch['status'] == 'complete'
            batch_responses = 0
            for lane in batch['lanes']:
                record = lane['record']
                key = record['task_id'], record['input_uid']
                slot = lane['repeat']
                assert key in records and (key, slot) not in seen
                seen.add((key, slot))
                assert lane['terminal']['original_slot_id'] == slot
                trajectory = lane['trajectory']
                n = len(trajectory['trials'])
                assert 1 <= n <= 3 and trajectory['terminal_trial'] == n - 1
                count = Counter(proposals=n)
                for t, trial in enumerate(trajectory['trials']):
                    assert trial['trial_index'] == t
                    assert (trial['reflection'] is not None) == (t < n - 1)
                    for role in ['thought', 'action', 'reflection']:
                        response = trial[role]
                        if response is None:
                            continue
                        identity = response['call_path'], response['batch_row']
                        assert identity not in identities, identity
                        identities.add(identity)
                        count['responses'] += 1
                        count['tokens'] += token_count(response)
                assert count['responses'] == trajectory['physical_calls'] == 3 * n - 1
                totals.update(count)
                byinput.setdefault(key, Counter()).update(count)
                batch_responses += count['responses']
            assert batch_responses == batch['logical_calls']
        print('REFLEXION', model, unit['unit_id'], len(seen), flush=True)
    assert seen == {(key, slot) for key in records for slot in range(20)}
    assert totals['responses'] == len(identities)
    for (task, uid), count in byinput.items():
        PER_INPUT.append(dict(model=model, method_id='reflexion', task_id=task, input_uid=uid, **count))
    return totals, dict(trajectories=len(seen), unique_recorded_response_identities=len(identities),
                        basis='All committed thought, action and reflection responses across twenty max-three-trial trajectories; reused generations counted once')


def tot(model, records):
    totals, diagnostics = Counter(), Counter()
    for number, ((task, uid), record) in enumerate(records.items()):
        path = RUN / 'tot/runs' / model.lower() / task / ('%04d' % record['input_idx']) / 'complete.json'
        d = read(path)
        assert d['record'] == record and d['method_id'] == 'tot_lm_vote_b4_c20'
        assert len(d['candidates']) == len(d['children']) == 20
        assert [c['original_slot_id'] for c in d['candidates']] == list(range(20))
        provenance = d['provenance']
        prefix = read(Path(provenance['old_complete_path']).parent / 'initial_prefix.json', provenance['old_prefix_sha256'])
        initial = prefix['initial_provenance']['calls']
        assert len(initial) == len(prefix['initial_candidates']) == 5
        tokens = sum(token_count(response) for response in initial)
        count = Counter(proposals=5, responses=5, tokens=tokens)
        used_responses = set()
        calls = {}
        for child in d['children']:
            call_path, row = child['call_path'], child['batch_row']
            if call_path not in calls:
                calls[call_path] = read(call_path)
            response = calls[call_path]['outputs'][row]
            assert response['raw_text'] == child['raw_text']
            assert (call_path, row) not in used_responses
            used_responses.add((call_path, row))
            count.update(proposals=1, responses=1, tokens=token_count(response))
        for path_field, hash_field in [('reused_parent_call_path', 'reused_parent_call_sha256'),
                                       ('terminal_vote_call_path', 'terminal_vote_sha256')]:
            call = read(provenance[path_field], provenance[hash_field])
            assert call['status'] == 'complete' and len(call['outputs']) == 5
            count.update(responses=5, tokens=sum(token_count(response) for response in call['outputs']))
        assert count['proposals'] == 25 and count['responses'] == d['costs']['logical_model_responses_total'] == 35
        for stage in ['parent_vote', 'terminal_vote']:
            diagnostics[stage + '_all_invalid_inputs'] += int(d[stage]['all_invalid_fallback'])
            diagnostics[stage + '_valid_votes'] += d[stage]['valid_votes']
        diagnostics['historical_children_outside_current_graph'] += 15 - d['costs']['reused_child_responses']
        totals.update(count)
        PER_INPUT.append(dict(model=model, method_id='tot', task_id=task, input_uid=uid, **count))
        if number % 500 == 0:
            print('TOT', model, number + 1, 'inputs', flush=True)
    assert totals['proposals'] == 3905 * 25 and totals['responses'] == 3905 * 35
    return totals, dict(diagnostics, basis='Five initial proposals, twenty children and two five-vote stages; original token costs of reused responses included once; historical branches outside this configuration excluded')


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Full trace accounting requires a CPU allocation'
    complete = read(RUN / 'results/completion.json')
    assert complete['status'] == 'COMPLETE_18_CONFIGURATIONS'
    records = {}
    for task in TASKS:
        path = BENCH / 'manifests' / (task + '.test.seen.jsonl')
        raw = path.read_bytes(); SOURCES[str(path)] = hashlib.sha256(raw).hexdigest()
        for line in raw.decode().splitlines():
            r = json.loads(line); records[task, r['input_uid']] = r
    assert len(records) == 3905
    baseline = ROOT / 'baseline_data/table2_k20_workload.csv'
    SOURCES[str(baseline)] = hashlib.sha256(baseline.read_bytes()).hexdigest()
    with baseline.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    details = {}
    for model in ['Mistral', 'Llama']:
        for method, run in [('tot', tot), ('reflexion', reflexion)]:
            counts, detail = run(model, records)
            row = next(r for r in rows if (r['model'], r['method_id']) == (model, method))
            assert row['status'] == 'full_run_pending'
            for field, key in [('molecular_proposals', 'proposals'), ('response_sequences', 'responses'), ('output_tokens', 'tokens')]:
                row[field + '_total'] = counts[key]
                row[field + '_mean'] = counts[key] / 3905
            row['status'] = 'complete_generation'
            row['basis'] = detail['basis']
            details[model + '/' + method] = dict(counts, **detail)
            print('COMPLETE WORKLOAD', model, method, dict(counts), flush=True)
    assert len(rows) == 18 and all(r['status'] == 'complete_generation' for r in rows)
    assert len(PER_INPUT) == 4 * 3905
    with (DATA / 'table2_k20_workload.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with gzip.open(ROOT / 'new_workload_per_input.jsonl.gz', 'wt') as stream:
        for row in PER_INPUT:
            stream.write(json.dumps(row) + '\n')
    source_path = ROOT / 'workload_source_index.json'
    source_path.write_text(json.dumps(SOURCES, indent=2) + '\n')
    audit = dict(status='PASS_COMPLETE_18_CONFIGURATION_WORKLOAD', complete_configurations=18,
                 preserved_previously_complete_rows=14, newly_counted_configurations=4,
                 reused_prefixes_included=True, physical_calls_not_conflated_with_sequences=True,
                 excludes='prompt processing, property computation, unreturned beam work, model loading, discarded attempts and historical branches outside the evaluated configuration',
                 sources_count=len(SOURCES), source_index_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                 new_details=details, generation_calls=0, property_prediction_calls=0)
    (DATA / 'k20_workload_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    main()
