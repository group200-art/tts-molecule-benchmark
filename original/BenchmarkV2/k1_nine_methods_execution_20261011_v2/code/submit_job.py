"""Record exact Slurm submissions for the user-authorized isolated execution."""
import argparse
import hashlib
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/work/users/w/a/wanzhou/conda_envs/gellmo_mistral_gen/bin/python'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--script', required=True)
    p.add_argument('--array')
    p.add_argument('--dependency')
    p.add_argument('--gpu', action='store_true')
    p.add_argument('--minutes', type=int, default=120)
    p.add_argument('--mem', default='32G')
    p.add_argument('--cpus', type=int, default=8)
    p.add_argument('args', nargs=argparse.REMAINDER)
    a = p.parse_args()
    assert a.name.replace('_', '').isalnum()
    script = (ROOT / a.script).resolve()
    assert ROOT in script.parents and script.is_file()
    auth = json.loads((ROOT / 'control/authorization.json').read_text())
    assert auth['status'] == 'AUTHORIZED_IMPLEMENTATION_AND_EXECUTION'
    receipt = ROOT / 'control' / (a.name + '_submission.json')
    assert not receipt.exists(), 'Already submitted or attempted; use its job ID rather than duplicating.'
    extra = a.args[1:] if a.args[:1] == ['--'] else a.args
    command = [PYTHON, '-B', str(script)] + extra
    body = ['#!/bin/bash', 'set -euo pipefail', 'unset PYTHONPATH PYTHONHOME',
            'export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1',
            'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1',
            'export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=' + str(a.cpus),
            'export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1']
    body += ['runtime_cache=' + shlex.quote(str(ROOT / 'runs/runtime_cache')) + '/"${SLURM_JOB_ID}"',
             'mkdir -p "$runtime_cache"']
    for variable, leaf in [('TMPDIR','tmp'), ('HF_HOME','hf'), ('HF_HUB_CACHE','hf/hub'),
                           ('XDG_CACHE_HOME','xdg'), ('TORCH_HOME','torch'), ('MPLCONFIGDIR','mpl'),
                           ('TRITON_CACHE_DIR','triton'), ('CUDA_CACHE_PATH','cuda'),
                           ('NUMBA_CACHE_DIR','numba')]:
        body += ['mkdir -p "$runtime_cache/' + leaf + '"',
                 'export ' + variable + '="$runtime_cache/' + leaf + '"']
    if not a.gpu:
        body += ["export CUDA_VISIBLE_DEVICES=''" ]
    body += ['exec ' + shlex.join(command), '']
    batch = ROOT / 'control' / (a.name + '.sbatch')
    with batch.open('x') as f:
        f.write('\n'.join(body))
    cmd = ['sbatch', '--parsable', '--job-name=' + a.name, '--account=rc_hzh_pi',
           '--partition=' + ('l40-gpu' if a.gpu else 'general'),
           '--qos=' + ('gpu_access' if a.gpu else 'normal'), '--nodes=1', '--ntasks=1',
           '--cpus-per-task=' + str(a.cpus), '--mem=' + a.mem, '--time=' + str(a.minutes),
           '--no-requeue', '--mail-type=NONE', '--chdir=' + str(ROOT),
           '--output=' + str(ROOT / 'logs' / (a.name + '_%A_%a.out')),
           '--error=' + str(ROOT / 'logs' / (a.name + '_%A_%a.err')),
           '--kill-on-invalid-dep=yes']
    if a.gpu:
        cmd += ['--gres=gpu:nvidia_l40s:1']
    if a.array:
        if a.gpu:
            assert a.array.endswith('%2'), 'Global GPU stage must limit concurrency to two.'
        cmd += ['--array=' + a.array]
    if a.dependency:
        cmd += ['--dependency=' + a.dependency]
    cmd.append(str(batch))
    result = subprocess.run(cmd, text=True, capture_output=True)
    data = dict(utc=datetime.now(timezone.utc).isoformat(), command=cmd,
                returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
                script_sha256=hashlib.sha256(batch.read_bytes()).hexdigest())
    if result.returncode == 0:
        data['job_id'] = result.stdout.strip().split(';')[0]
        assert data['job_id'].isdigit()
    with receipt.open('x') as f:
        json.dump(data, f, indent=2)
    print(json.dumps(data), flush=True)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
