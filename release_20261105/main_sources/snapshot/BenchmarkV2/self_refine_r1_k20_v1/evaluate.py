"""Evaluate each completed seed on CPU; resume committed blocks without resampling."""
import argparse
import fcntl
import os
import subprocess
import time
import traceback
from common import *
from property_bridge import requests_for


def child(role, env_name, args, directory, name):
    # Attempts use fresh logs, while successfully published scientific outputs persist.
    attempt = directory / 'attempts' / os.environ['SLURM_JOB_ID']
    attempt.mkdir(parents=True, exist_ok=True)
    command = [ENVS[env_name], '-B', str(ROOT / 'property_worker.py'), role] + [str(a) for a in args]
    started = time.perf_counter()
    with (attempt / (name + '.stdout')).open('x') as out, (attempt / (name + '.stderr')).open('x') as err:
        result = subprocess.run(command, stdout=out, stderr=err, env=isolate(attempt, cpu=True), cwd=str(directory))
    save(attempt / (name + '.process.json'), dict(argv=command, returncode=result.returncode,
         wall_seconds=time.perf_counter()-started))
    assert result.returncode == 0, 'Property subprocess failed: ' + str(attempt / (name + '.stderr'))


def block(candidates, task, seed, number):
    directory = ROOT / 'evaluation' / str(seed) / task / ('block{:03d}'.format(number))
    directory.mkdir(parents=True, exist_ok=True)
    identity = digest(candidates)
    done = directory / 'complete.json'
    if done.exists():
        status = read(done)
        assert status['candidate_digest'] == identity
        assert sha(directory / 'slots_result.json') == status['slots_sha256']
        return read(directory / 'slots_result.json')['rows']
    spec = read(LEGACY / 'configs/properties.json')['tasks'][task]
    requests, mapping = requests_for(candidates, spec['cache_contract'])
    assert len(requests) <= 80
    inp = directory / 'requests.json'
    payload = dict(task_id=task, requests=requests, admet_columns=spec['admet_columns'])
    if inp.exists():
        assert read(inp) == payload
    else:
        save(inp, payload)
    # Incomplete subprocess directories are preserved; retry in a fresh attempt root.
    attempt = directory / 'work' / os.environ['SLURM_JOB_ID']
    attempt.mkdir(parents=True, exist_ok=True)
    admet_receipt = directory / 'admet_receipt.json'
    if not admet_receipt.exists():
        dest = attempt / 'admet'
        child('admet', 'gellmo_inf', ['--requests', inp, '--output', dest], directory, 'admet')
        save(admet_receipt, dict(path=str(dest / 'admet_result.json'), sha256=sha(dest / 'admet_result.json')))
    ar = read(admet_receipt)
    assert sha(ar['path']) == ar['sha256'] and read(ar['path'])['requests_sha256'] == sha(inp)
    standard_receipt = directory / 'standard_receipt.json'
    if not standard_receipt.exists():
        dest = attempt / 'standard'
        child('standard', 'gellmo_eval_py36', ['--mode', 'produce', '--input', inp, '--input-sha', sha(inp),
              '--admet', ar['path'], '--output', dest], directory, 'standard')
        save(standard_receipt, dict(path=str(dest / 'responses_raw.json'), sha256=sha(dest / 'responses_raw.json')))
    sr = read(standard_receipt)
    assert sha(sr['path']) == sr['sha256']
    responses = read(sr['path'])
    assert len(responses) == len(requests)
    bridge = dict(requests=requests, mapping=mapping, responses=responses, cache_contract=spec['cache_contract'])
    slot_input = directory / 'slots_input.json'
    slot_payload = dict(records=manifest(task), candidates=candidates, bridge=bridge)
    if slot_input.exists():
        assert read(slot_input) == slot_payload
    else:
        save(slot_input, slot_payload)
    out = directory / 'slots_result.json'
    if not out.exists():
        child('standard', 'gellmo_eval_py36', ['--mode', 'slots', '--input', slot_input,
              '--input-sha', sha(slot_input), '--output', out], directory, 'slots')
    rows = read(out)['rows']
    assert [r['candidate_id'] for r in rows] == [c['candidate_id'] for c in candidates]
    assert all(r['r'] == 1 and r['rep_seed'] == seed for r in rows)
    save(done, dict(status='complete', candidate_digest=identity, slots_sha256=sha(out), n=len(rows)))
    print(json.dumps(dict(event='property_block_complete', task=task, seed=seed, block=number, n=len(rows))), flush=True)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--index', type=int, default=int(os.environ.get('SLURM_ARRAY_TASK_ID', '0')))
    p.add_argument('--probe', action='store_true')
    args = p.parse_args()
    require_allocation()
    verify()
    assert 0 <= args.index < 17
    seed = 51004 + args.index
    directory = ROOT / 'evaluation' / str(seed)
    directory.mkdir(parents=True, exist_ok=True)
    attempt = directory / 'attempts' / os.environ['SLURM_JOB_ID']
    attempt.mkdir(parents=True, exist_ok=False)
    os.environ.update(isolate(attempt, cpu=True))
    with (directory / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.perf_counter()
        status = dict(status='running', seed=seed, completed_candidates=0, job_id=os.environ['SLURM_JOB_ID'])
        save(attempt / 'started.json', status)
        try:
            candidates = generated_candidates(seed, allow_partial=args.probe)
            if args.probe:
                assert seed == 51004 and len(candidates) == 80
            for task in TASKS:
                items = [c for c in candidates if c['task_id'] == task]
                if not items:
                    continue
                assert [c['input_idx'] for c in items] == list(range(len(items)))
                rows = []
                for start in range(0, len(items), 80):
                    rows.extend(block(items[start:start+80], task, seed, start // 80))
                    status['completed_candidates'] += len(items[start:start+80])
                    save(directory / 'progress.json', status, replace=True)
                if not args.probe:
                    assert len(rows) == len(manifest(task))
                    destination = directory / task / 'result.json.gz'
                    if not destination.exists():
                        save(destination, dict(status='complete', task_id=task, rep_seed=seed, r=1, rows=rows))
            if args.probe:
                status.update(status='PASS', gate='structure, identity and successful property pipeline only; no performance threshold',
                              new_trajectories_reusable=80, property_candidates=80)
                save(ROOT / 'probe_gate.json', status)
            else:
                assert status['completed_candidates'] == 3905
                status['status'] = 'complete'
                if not (directory / 'complete.json').exists():
                    save(directory / 'complete.json', status)
        except BaseException as error:
            status.update(status='failed', exception=repr(error), traceback=traceback.format_exc())
            raise
        finally:
            status['attempt_wall_seconds'] = time.perf_counter() - started
            save(attempt / 'status.json', status)
            print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
