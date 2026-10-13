"""At most two concurrent GPUs across all three methods; fail-closed aggregation."""
import argparse
import shlex
import subprocess
from common import *


def submit(name, script, args, gpu=False, array=None, dependency=None, hours=2, cpus=None):
    (ROOT / 'logs').mkdir(exist_ok=True)
    (ROOT / 'jobs').mkdir(exist_ok=True)
    command = [ENVS['gellmo_mistral_gen'], '-B', str(ROOT / script)] + args
    body = '\n'.join(['#!/bin/bash', 'set -euo pipefail', 'unset PYTHONPATH PYTHONHOME',
        'export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1',
        'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1',
        'export THREE_K20_EXECUTION=' + shlex.quote(EXECUTION), 'exec ' + shlex.join(command), ''])
    path = ROOT / 'jobs' / (name + '.sbatch')
    with path.open('x') as f:
        f.write(body)
    cmd = ['sbatch', '--parsable', '--job-name=' + name, '--account=rc_hzh_pi',
        '--partition=' + ('hzhlab' if gpu else 'general'), '--qos=normal', '--nodes=1', '--ntasks=1',
        '--cpus-per-task=' + str(cpus or (8 if gpu else 4)), '--mem=' + ('64G' if gpu else '16G'),
        '--time={:02d}:00:00'.format(hours), '--no-requeue', '--mail-type=NONE',
        '--chdir=' + str(ROOT), '--output=' + str(ROOT / 'logs' / (name + '_%A_%a.out')),
        '--error=' + str(ROOT / 'logs' / (name + '_%A_%a.err')), '--kill-on-invalid-dep=yes']
    if gpu:
        cmd.append('--gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:1')
    if array:
        cmd.append('--array=' + array)
    if dependency:
        cmd.append('--dependency=' + dependency)
    cmd.append(str(path))
    result = subprocess.run(cmd, capture_output=True, text=True)
    receipt = dict(argv=cmd, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
                   script_sha256=sha(path))
    save(ROOT / 'jobs' / (name + '_submission.json'), receipt)
    assert result.returncode == 0, result.stderr
    job = result.stdout.strip().split(';')[0]
    assert job.isdigit(), result.stdout
    print(name, job, flush=True)
    return job


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['probe', 'production', 'queue-production'])
    args = p.parse_args()
    verify()
    assert read(ROOT / 'cpu_checks.json')['status'] == 'PASS'
    if args.stage == 'probe':
        array = ','.join(map(str, read(ROOT / 'probe_units.json'))) + '%2'
        job = submit('tts3_k20_probe', 'worker.py', ['--max-batches', '1'], gpu=True, array=array)
        save(ROOT / 'probe_submission.json', dict(job_array=job, max_concurrent_gpus=2))
    else:
        probe = read(ROOT / 'probe_submission.json')['job_array']
        if args.stage == 'queue-production':
            gate = submit('tts3_k20_gate', 'gate.py', [], dependency='afterok:' + probe)
            dependency = 'afterok:' + gate
        else:
            assert read(ROOT / 'probe_gate.json')['status'] == 'PASS'
            gate = None
            dependency = 'afterok:' + probe
        job = submit('tts3_k20_full', 'worker.py', [], gpu=True, array='0-167%2',
                     dependency=dependency, hours=6)
        summary = submit('tts3_k20_table1', 'aggregate.py', [], dependency='afterok:' + job, hours=2)
        save(ROOT / 'production_submission.json', dict(job_array=job, aggregation=summary, gate=gate,
             max_concurrent_gpus=2, inputs_per_method=3905, slots_per_method=78100,
             methods=METHODS, reused_probe_slots=720, total_new_method_slots=234300))


if __name__ == '__main__':
    main()
