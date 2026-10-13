"""One bounded full-manifest unit; probes commit the first formal batch for reuse."""
import argparse
import fcntl
import time
import traceback
from common import *
from replay import Replay, next_group, group_seed
from batched_generation import BatchedGeneration
from property_client import PropertyClient
from metrics_entry import evaluate_registered, aggregate_registered


def reflexion_terminals(batch, generated, directory):
    """Reuse the terminal trial's measured properties, including its exact round2 values."""
    if (directory / 'complete.json').exists():
        receipt = read(directory / 'complete.json')
        assert receipt['generation_sha256'] == sha(directory.parent / 'generation.json.gz')
        return read(receipt['result_path'])
    filled = []
    for lane in generated['lanes']:
        final = lane['trajectory']['trials'][-1]['evaluation']['filled_candidate']
        terminal = lane['terminal']
        assert final['candidate_id'] == lane['trajectory']['native_candidate_id']
        assert final['raw_smiles'] == terminal['raw_smiles'] and final['source_smiles'] == terminal['source_smiles']
        item = dict(final, **terminal)
        item['source_trial_candidate_id'] = final['candidate_id']
        filled.append(item)
    rows = []
    for record in batch['records']:
        pool = [c for c in filled if c['input_uid'] == record['input_uid']]
        row = evaluate_registered(record['task_id'], record['input_uid'], record['source_properties'], pool, 20)
        native = [lane['trajectory']['trials'][-1]['feedback']['success'] for lane in generated['lanes']
                  if lane['record']['input_uid'] == record['input_uid']]
        assert row['SR_dir_success'] == any(native)
        rows.append(row)
    result = dict(status='complete', filled=filled, rows=rows, metrics=aggregate_registered(rows),
                  property_basis='Exact terminal trial raw and round2 properties; no second predictor call')
    path = directory / 'result.json.gz'
    save(path, result)
    save(directory / 'complete.json', dict(status='complete', result_path=str(path),
         generation_sha256=sha(directory.parent / 'generation.json.gz'), extra_property_predictions=0,
         plan_sha256=sha(ROOT / 'plan.json')))
    return result


def run_batch(batch, model, properties, directory):
    complete = directory / 'generation.json.gz'
    if complete.exists():
        result = read(complete)
        assert result['batch'] == batch and result['plan_sha256'] == sha(ROOT / 'plan.json')
        return result
    directory.mkdir(parents=True, exist_ok=True)
    replay = Replay(batch, model.tok)
    physical = []
    for path in sorted((directory / 'calls').glob('*.json.gz')):
        result = read(path)
        assert result['plan_sha256'] == sha(ROOT / 'plan.json') and result['batch_id'] == batch['batch_id']
        requests = [o['request'] for o in result['outputs']]
        assert result['seed'] == group_seed(batch, requests)
        for o in result['outputs']:
            assert o['request']['key'] not in replay.calls
            replay.calls[o['request']['key']] = o
        physical.append(result)
    evaluation_count = 0
    for path in sorted((directory / 'feedback').glob('eval_*/complete.json')):
        result = read(read(path)['result_path'])
        for item in result['individual']:
            assert item['candidate']['candidate_id'] not in replay.evaluations
            replay.evaluations[item['candidate']['candidate_id']] = item
        evaluation_count += 1
    started = time.time()
    while True:
        done, calls, evaluations = replay.pending()
        if len(done) == len(batch['lanes']):
            break
        if evaluations:
            subset = list({item['record']['input_uid']: item['record'] for item in evaluations}.values())
            result = properties.evaluate(batch['method'], batch['task_id'], subset,
                [item['candidate'] for item in evaluations], 'feedback',
                directory / 'feedback' / ('eval_{:02d}'.format(evaluation_count)))
            for item in result['individual']:
                replay.evaluations[item['candidate']['candidate_id']] = item
            evaluation_count += 1
        else:
            assert calls
            requests = next_group(calls)
            index = len(physical)
            path = directory / 'calls' / ('{:03d}_{}_{}.json.gz'.format(index, os.environ['SLURM_JOB_ID'], os.getpid()))
            result = model.run(batch, requests, index, path)
            for item in result['outputs']:
                assert item['request']['key'] not in replay.calls
                replay.calls[item['request']['key']] = item
            physical.append(result)
    assert [x['lane_id'] for x in done] == [x['lane_id'] for x in batch['lanes']]
    for r in batch['records']:
        slots = [x['terminal']['original_slot_id'] for x in done if x['record']['input_uid'] == r['input_uid']]
        assert slots == list(range(20))
    result = dict(status='complete', batch=batch, lanes=done, plan_sha256=sha(ROOT / 'plan.json'),
                  physical_calls=len(physical), logical_calls=len(replay.calls),
                  physical_cuda_seconds=sum(p['physical_cuda_wall_seconds'] for p in physical),
                  feedback_batches=evaluation_count, feedback_candidates=len(replay.evaluations),
                  generation_and_feedback_wall_seconds_this_attempt=time.time() - started)
    save(complete, result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--index', type=int)
    p.add_argument('--max-batches', type=int)
    args = p.parse_args()
    require_allocation()
    verify()
    if not args.max_batches:
        assert read(ROOT / 'probe_gate.json')['status'] == 'PASS', 'Full production requires engineering gate'
    index = args.index if args.index is not None else int(os.environ['SLURM_ARRAY_TASK_ID'])
    unit = units()[index]
    assert unit['index'] == index
    directory = ROOT / 'runs' / unit['unit_id']
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / '.worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (directory / 'complete.json').exists():
        assert read(directory / 'complete.json')['plan_sha256'] == sha(ROOT / 'plan.json')
        print('Already complete:', unit['unit_id'], flush=True)
        return
    attempt = directory / ('worker_{}_{}'.format(os.environ['SLURM_JOB_ID'], os.getpid()))
    attempt.mkdir(exist_ok=False)
    os.environ.update(isolate(attempt / 'environment'))
    client = PropertyClient(attempt)
    started = time.time()
    save(attempt / 'started.json', dict(unit=unit, unix=started, job=os.environ['SLURM_JOB_ID'],
                                     probe=bool(args.max_batches), plan_sha256=sha(ROOT / 'plan.json')))
    try:
        selected = list(batches(unit))
        if args.max_batches:
            selected = selected[:args.max_batches]
        model = BatchedGeneration(unit['method'], attempt)
        if any(not (directory / b['batch_id'] / 'generation.json.gz').exists() for b in selected):
            model.start()
        generated = []
        for batch in selected:
            result = run_batch(batch, model, client, directory / batch['batch_id'])
            generated.append(result)
            save(directory / 'progress.json', dict(unit=unit['unit_id'], stage='generation',
                 batches_done=len(generated), batches_selected=len(selected), unix=time.time()), replace=True)
        # Offline scoring for s1/ICL starts only after selected generation has completed.
        rows = []
        for batch, result in zip(selected, generated):
            if unit['method'] == 'reflexion':
                scored = reflexion_terminals(batch, result, directory / batch['batch_id'] / 'terminals')
            else:
                scored = client.evaluate(unit['method'], unit['task_id'], batch['records'],
                    [x['terminal'] for x in result['lanes']], 'terminals', directory / batch['batch_id'] / 'terminals',
                    cache_check=bool(args.max_batches) and index == 0)
            rows.extend(scored['rows'])
        assert len(rows) == sum(len(b['records']) for b in selected)
        result = dict(status='complete', unit=unit, probe=bool(args.max_batches), rows=rows,
                      inputs=len(rows), slots=sum(r['n_slots'] for r in rows),
                      batches=len(selected), physical_cuda_seconds=sum(b['physical_cuda_seconds'] for b in generated),
                      logical_calls=sum(b['logical_calls'] for b in generated),
                      feedback_candidates=sum(b['feedback_candidates'] for b in generated),
                      model_load_seconds=getattr(model, 'load_seconds', 0),
                      wall_seconds=time.time() - started, completed_unix=time.time(),
                      plan_sha256=sha(ROOT / 'plan.json'))
        save(attempt / 'completed.json', {k:v for k,v in result.items() if k != 'rows'})
        if args.max_batches:
            save(directory / 'probe.json.gz', result)
        else:
            assert len(rows) == unit['stop'] - unit['start'] and result['slots'] == 20 * len(rows)
            save(directory / 'result.json.gz', result)
            save(directory / 'complete.json', {k:v for k,v in result.items() if k != 'rows'})
        print(json.dumps({k:v for k,v in result.items() if k not in ('rows', 'unit')}), flush=True)
    except BaseException:
        save(attempt / 'failed.json', dict(error=traceback.format_exc(), unix=time.time()))
        raise
    finally:
        client.close()
        lock.close()


if __name__ == '__main__':
    main()
