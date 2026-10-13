"""Bounded allocated execution; no changes to shared Python environments."""
import os
import subprocess
import time
from pathlib import Path
from contracts import ROOT, APPROVAL, read, write, verify, unit_spec

ENVS = {name: '/work/users/w/a/wanzhou/conda_envs/' + name + '/bin/python'
        for name in ('gellmo_mistral_gen', 'gellmo_inf', 'gellmo_eval_py36')}


def require_allocation(expected_unit=None):
    if os.environ.get('K20_EXECUTION') != APPROVAL:
        raise PermissionError('K20 execution identity missing')
    for key in ('SLURM_JOB_ID', 'SLURM_ARRAY_JOB_ID', 'SLURM_ARRAY_TASK_ID'):
        if not os.environ.get(key, '').isdigit():
            raise PermissionError('An allocated Slurm array is required')
    if os.environ.get('SLURM_RESTART_COUNT', '0') != '0':
        raise PermissionError('Automatic requeue is disabled')
    offset = int(os.environ.get('K20_UNIT_OFFSET', '0'))
    if offset < 0:
        raise ValueError('Invalid unit offset')
    unit = unit_spec(offset + int(os.environ['SLURM_ARRAY_TASK_ID']))
    if expected_unit is not None and expected_unit != unit['unit_id']:
        raise ValueError('Unit identity mismatch')
    return unit


def isolated_env(directory, cpu=False):
    directory = Path(directory)
    assert ROOT in directory.resolve().parents
    env = dict(os.environ)
    for key in ('PYTHONPATH', 'PYTHONHOME'):
        env.pop(key, None)
    for key in ('TMPDIR', 'TMP', 'TEMP', 'HF_HOME', 'HF_HUB_CACHE', 'TRANSFORMERS_CACHE',
                'XDG_CACHE_HOME', 'TORCH_HOME', 'MPLCONFIGDIR', 'TRITON_CACHE_DIR',
                'CUDA_CACHE_PATH', 'NUMBA_CACHE_DIR', 'JOBLIB_TEMP_FOLDER'):
        folder = directory / 'cache' / key.lower()
        folder.mkdir(parents=True, exist_ok=True)
        env[key] = str(folder)
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1', USE_TORCH='1', USE_TF='0',
               USE_FLAX='0', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='8',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    if cpu:
        env['CUDA_VISIBLE_DEVICES'] = ''
    return env


def checked_child(interpreter, script, arguments, directory, label, costs):
    argv = [interpreter, '-B', str(ROOT / script)] + list(map(str, arguments))
    item = dict(label=label, argv=argv, completed=False)
    costs.append(item)
    write(directory / (label + '_invocation.json'), item)
    start = time.perf_counter()
    with (directory / (label + '.stdout')).open('x') as out, (directory / (label + '.stderr')).open('x') as err:
        result = subprocess.run(argv, stdout=out, stderr=err, env=isolated_env(directory, cpu=True), cwd=str(directory))
    item.update(returncode=result.returncode, completed=result.returncode == 0, wall_seconds=time.perf_counter() - start)
    write(directory / (label + '_process.json'), item)
    if result.returncode:
        raise RuntimeError('Property/evaluation process failed: ' + label)


def verify_code_inventory():
    for item in read(ROOT / 'evidence/runtime_inventory.json'):
        verify(item['path'], item['sha256'])
