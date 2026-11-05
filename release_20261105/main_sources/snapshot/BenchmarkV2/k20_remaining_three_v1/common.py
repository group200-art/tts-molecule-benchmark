"""Full3905 K20 continuation: isolated outputs and immutable source contracts."""
import gzip
import hashlib
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
LEGACY = REPO / 'BenchmarkV2/k20_alignment_20261002_v1'
FROZEN = REPO / 'BenchmarkV2/table1_k20_frozen_20261003_v1'
sys.path.insert(0, str(LEGACY / 'src'))
from contracts import digest, manifest, task_contract

TASKS = ['BDP', 'BDQ', 'BPQ', 'DPQ', 'BDPQ', 'MPQ', 'BDMQ', 'BHMQ', 'BMPQ', 'HMPQ']
METHODS = ['fewshot', 's1', 'reflexion']
EXECUTION = 'BV2-REMAINING-THREE-K20-v1'
ENVS = {n: '/ENVIRONMENT_ROOT/' + n + '/bin/python'
        for n in ('gellmo_mistral_gen', 'gellmo_inf', 'gellmo_eval_py36')}


def read(path):
    path = Path(path)
    with (gzip.open if path.suffix == '.gz' else open)(str(path), 'rt', encoding='utf-8') as f:
        return json.load(f)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def destination(path):
    path = Path(path)
    assert path.is_absolute() and ROOT in path.resolve().parents and path == path.resolve(), str(path)
    return path


def save(path, value, replace=False):
    path = destination(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.tmp.' + str(os.getpid()))
    payload = (json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n').encode()
    with tmp.open('xb') as f:
        if path.suffix == '.gz':
            with gzip.GzipFile(fileobj=f, mode='wb', mtime=0) as z:
                z.write(payload)
        else:
            f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    if replace:
        os.replace(str(tmp), str(path))
    else:
        os.link(str(tmp), str(path))
        tmp.unlink()


def require_allocation():
    assert os.environ.get('SLURM_JOB_ID'), 'Heavy work requires Slurm allocation'
    assert os.environ.get('THREE_K20_EXECUTION') == EXECUTION
    assert os.environ.get('SLURM_RESTART_COUNT', '0') == '0'


def isolate(directory, cpu=False):
    directory = destination(directory)
    env = dict(os.environ)
    for key in ('PYTHONPATH', 'PYTHONHOME'):
        env.pop(key, None)
    for key, leaf in {'TMPDIR': 'tmp', 'HF_HOME': 'cache/hf', 'HF_HUB_CACHE': 'cache/hf/hub',
                      'TRANSFORMERS_CACHE': 'cache/hf/transformers', 'XDG_CACHE_HOME': 'cache/xdg',
                      'TORCH_HOME': 'cache/torch', 'MPLCONFIGDIR': 'cache/mpl',
                      'TRITON_CACHE_DIR': 'cache/triton', 'CUDA_CACHE_PATH': 'cache/cuda',
                      'NUMBA_CACHE_DIR': 'cache/numba'}.items():
        p = directory / leaf
        p.mkdir(parents=True, exist_ok=True)
        env[key] = str(p)
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='8', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', USE_TORCH='1', USE_TF='0', USE_FLAX='0')
    if cpu:
        env['CUDA_VISIBLE_DEVICES'] = ''
    return env


@lru_cache(None)
def plan():
    return read(ROOT / 'plan.json')


def units():
    return read(ROOT / 'units.json')


def verify():
    for entry in read(ROOT / 'inventory.json'):
        assert sha(entry['path']) == entry['sha256'], entry['path']
    evidence = REPO / 'BenchmarkV2/method_review_P5_v1/execution_v1/evidence'
    for name in ('model_metadata_before.json', 'property_metadata_before.json'):
        for entry in read(evidence / name):
            st = Path(entry['path']).stat()
            assert st.st_size == entry['size'] and st.st_mtime_ns == entry['mtime_ns'], entry['path']


def records(unit):
    result = manifest(unit['task_id'])[unit['start']:unit['stop']]
    assert digest(result) == unit['records_sha256']
    return result


def batches(unit):
    rows = records(unit)
    for start in range(0, len(rows), 4):
        part = rows[start:start + 4]
        yield dict(batch_id=unit['unit_id'] + '_b{:02d}'.format(start // 4),
                   method=unit['method'], task_id=unit['task_id'], records=part,
                   lanes=[dict(lane_id=r['input_uid'] + '_r{:02d}'.format(repeat), record=r, repeat=repeat)
                          for r in part for repeat in range(20)])


@lru_cache(None)
def property_spec(method, task):
    name = {'s1': 'properties.json', 'fewshot': 'properties_fewshot.json',
            'reflexion': 'properties_reflexion.json'}[method]
    return read(LEGACY / 'configs' / name)['tasks'][task]


def terminal(method, lane, raw_smiles, raw_text, trace):
    r = lane['record']
    ident = dict(method_id=plan()['extensions'][method]['main_id'], model_id='gellmo_p6_mistral',
                 task_id=r['task_id'], input_uid=r['input_uid'], original_slot_id=lane['repeat'],
                 execution=EXECUTION)
    return dict(ident, candidate_id='k20-terminal-' + digest(ident), input_idx=r['input_idx'],
                source_smiles=r['source_smiles'], raw_smiles=raw_smiles, raw_text=raw_text,
                trace=trace)
