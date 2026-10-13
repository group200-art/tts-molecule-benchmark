"""Bounded Slurm chain: engineering gate, 17 GPU repeats, CPU evaluation, aggregation."""
import argparse
import shlex
import subprocess
from common import *


def submit(name, script, args, gpu=False, array=None, dependency=None, hours=1):
    (ROOT/'logs').mkdir(exist_ok=True)
    (ROOT/'jobs').mkdir(exist_ok=True)
    command = [ENVS['gellmo_mistral_gen'], '-B', str(ROOT/script)] + args
    body = '\n'.join(['#!/bin/bash', 'set -euo pipefail',
        'unset PYTHONPATH PYTHONHOME',
        'export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1',
        'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1',
        'export SR_K20_EXECUTION='+shlex.quote(EXECUTION),
        'exec '+shlex.join(command), ''])
    path = ROOT/'jobs'/(name+'.sbatch')
    with path.open('x') as f:
        f.write(body)
    cmd = ['sbatch', '--parsable', '--job-name='+name, '--account=rc_hzh_pi',
        '--partition='+('hzhlab' if gpu else 'general'), '--qos=normal', '--nodes=1', '--ntasks=1',
        '--cpus-per-task='+('8' if gpu else '4'), '--mem='+('64G' if gpu else '16G'),
        '--time={:02d}:00:00'.format(hours), '--no-requeue', '--mail-type=NONE',
        '--chdir='+str(ROOT), '--output='+str(ROOT/'logs'/(name+'_%A_%a.out')),
        '--error='+str(ROOT/'logs'/(name+'_%A_%a.err')), '--kill-on-invalid-dep=yes']
    if gpu:
        cmd.append('--gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:1')
    if array:
        cmd.append('--array='+array)
    if dependency:
        cmd.append('--dependency='+dependency)
    cmd.append(str(path))
    result = subprocess.run(cmd, capture_output=True, text=True)
    receipt = dict(argv=cmd, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
                   script_sha256=sha(path))
    save(ROOT/'jobs'/(name+'_submission.json'), receipt)
    assert result.returncode == 0, result.stderr
    job = result.stdout.strip().split(';')[0]
    assert job.isdigit(), result.stdout
    print(name, job, flush=True)
    return job


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['probe', 'production'])
    args = p.parse_args()
    verify()
    assert read(ROOT/'cpu_checks.json')['status'] == 'PASS'
    assert read(ROOT/'reuse_audit.json')['status'] == 'PASS'
    if args.stage == 'probe':
        g = submit('sr_r1_gate_gen', 'generate.py', ['--index', '0', '--max-batches', '1'], gpu=True)
        e = submit('sr_r1_gate_eval', 'evaluate.py', ['--index', '0', '--probe'], dependency='afterok:'+g, hours=2)
        save(ROOT/'probe_submission.json', dict(generation=g, evaluation=e))
    else:
        assert read(ROOT/'probe_gate.json')['status'] == 'PASS'
        g = submit('sr_r1_k20_gen', 'generate.py', [], gpu=True, array='0-16%2', hours=6)
        e = submit('sr_r1_k20_eval', 'evaluate.py', [], array='0-16%4', dependency='aftercorr:'+g, hours=12)
        a = submit('sr_r1_k20_summary', 'aggregate.py', [], dependency='afterok:'+e, hours=2)
        save(ROOT/'production_submission.json', dict(generation_array=g, evaluation_array=e, aggregation=a,
             max_concurrent_gpus=2, generation_slots=66385, reused_slots=11715))


if __name__ == '__main__':
    main()
