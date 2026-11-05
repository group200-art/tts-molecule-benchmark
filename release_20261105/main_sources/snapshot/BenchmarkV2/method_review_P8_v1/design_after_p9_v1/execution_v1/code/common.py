"""Private execution contracts; no legacy output paths or implicit retries."""
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[3]
EXECUTION_ID = 'BV2-P8-GWIDTH-THREE-BUDGETS-TEN-TASKS-RUN01-v1'
GEN_PY = '/ENVIRONMENT_ROOT/gellmo_mistral_gen/bin/python'
INF_PY = '/ENVIRONMENT_ROOT/gellmo_inf/bin/python'
EVAL_PY = '/ENVIRONMENT_ROOT/gellmo_eval_py36/bin/python'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                    sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()


def text_sha(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def destination(path):
    p = Path(os.path.abspath(str(path)))
    if ROOT not in p.parents or p.resolve() != p:
        raise PermissionError('Outside execution root or symlink: ' + str(p))
    return p


def write(path, value, replace=False):
    p = destination(path)
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n'
    p.parent.mkdir(parents=True, exist_ok=True)
    if not replace:
        with p.open('x', encoding='utf-8') as f:
            f.write(payload)
    else:
        temp = p.with_name(p.name + '.tmp.' + str(os.getpid()))
        with temp.open('x', encoding='utf-8') as f:
            f.write(payload)
            f.flush()
        os.replace(str(temp), str(p))


def append(path, value):
    p = destination(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')
        f.flush()


def require_allocation():
    if not os.environ.get('SLURM_JOB_ID'):
        raise PermissionError('Real execution requires authorized Slurm allocation')
    if os.environ.get('P8_EXECUTION_ID') != EXECUTION_ID:
        raise PermissionError('Wrong execution authorization')


def configure_environment(directory, cpu=False):
    directory = destination(directory)
    directory.mkdir(parents=True, exist_ok=True)
    settings = {'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1',
                'TRANSFORMERS_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
                'TOKENIZERS_PARALLELISM': 'false', 'WANDB_DISABLED': 'true', 'PYTHONNOUSERSITE': '1',
                'USE_TORCH': '1', 'USE_TF': '0', 'USE_FLAX': '0',
                'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8',
                'NUMEXPR_NUM_THREADS': '8', 'P8_EXECUTION_ID': EXECUTION_ID}
    threads = str(min(8, int(os.environ.get('SLURM_CPUS_PER_TASK', '4' if cpu else '8'))))
    for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        settings[key] = threads
    for key in ['MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        settings[key] = '1'
    for name in ['TMPDIR', 'TMP', 'TEMP', 'XDG_CACHE_HOME', 'HF_HOME', 'TORCH_HOME',
                 'MPLCONFIGDIR', 'TRITON_CACHE_DIR', 'NUMBA_CACHE_DIR', 'JOBLIB_TEMP_FOLDER',
                 'TRANSFORMERS_CACHE', 'HF_HUB_CACHE', 'HF_DATASETS_CACHE', 'CUDA_CACHE_PATH']:
        path = directory / name.lower()
        path.mkdir(exist_ok=True)
        settings[name] = str(path)
    if cpu:
        settings['CUDA_VISIBLE_DEVICES'] = ''
    os.environ.update(settings)
    os.environ.pop('PYTHONPATH', None)
    os.environ.pop('PYTHONHOME', None)
    return settings


CONTRACT = read(ROOT / 'configs/design_contract.json')


def task_contract(task_id, supplied=None):
    t = next(t for t in CONTRACT['tasks'] if t['task_id'] == task_id)
    if supplied is not None and supplied != t['objectives']:
        raise ValueError('Task objectives changed')
    return t


def kernel():
    import metrics_kernel
    return vars(metrics_kernel)


def manifest(task_id):
    t = task_contract(task_id)
    if sha(t['manifest_path']) != t['manifest_sha256']:
        raise ValueError('Manifest SHA mismatch')
    rows = [json.loads(s) for s in Path(t['manifest_path']).read_text().splitlines()]
    assert len(rows) == t['actual_n']
    assert [r['input_idx'] for r in rows] == list(range(len(rows)))
    assert len({r['input_uid'] for r in rows}) == len(rows)
    names = [o['property'] for o in t['objectives']]
    for r in rows:
        assert r['task_id'] == task_id and r['task_key'] == t['task_key']
        kernel()['require_source'](r['source_properties'], names)
    return rows


def unit(stage, index):
    hits = [u for u in CONTRACT['planned_logical_shards'] if u['stage'] == stage and u['array_index'] == index]
    assert len(hits) == 1
    return hits[0]


def unit_key(u):
    return '{}_{:03d}_{}_l{:02d}'.format(u['stage'], u['array_index'], u['task_id'], u['logical_shard'])


def unit_inputs(u):
    rows = manifest(u['task_id'])[u['input_idx_start']:u['input_idx_end_exclusive']]
    assert len(rows) == u['n_inputs']
    assert text_sha(''.join(r['input_uid'] + '\n' for r in rows)) == u['ordered_input_uid_digest']
    return rows


def verify_execution():
    a = read(ROOT / 'control/authorization.json')
    assert a['execution_approved'] and a['execution_id'] == EXECUTION_ID
    assert not (ROOT / 'control/authorization_closed.json').exists()
    for file, expected in a['design_sha256'].items():
        assert sha(ROOT.parent / file) == expected
    for r in read(ROOT / 'configs/implementation_manifest.json')['files']:
        assert sha(ROOT / r['path']) == r['sha256'], r['path']
    for r in read(ROOT / 'configs/asset_metadata.json')['files']:
        stat = Path(r['path']).stat()
        assert stat.st_size == r['size'] and stat.st_mtime_ns == r['mtime_ns'], r['path']


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
