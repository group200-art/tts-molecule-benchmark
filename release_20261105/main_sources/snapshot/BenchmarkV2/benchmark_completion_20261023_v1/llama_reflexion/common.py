"""Isolated Llama R20 scope; native scientific sources remain read-only."""
import gzip
import hashlib
import importlib.util
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
LEGACY = REPO / 'BenchmarkV2/p6_llama_main_ten_tasks_v1'
ENGINE = REPO / 'BenchmarkV2/k20_remaining_three_v1'
S1 = REPO / 'BenchmarkV2/benchmark_paired_k1_k20_20261014_v1/llama_s1'
sys.path.insert(0, str(LEGACY / 'src'))
sys.path.append(str(ENGINE))
from contracts import digest, manifest, task_contract

TASKS = ['BDP', 'BDQ', 'BPQ', 'DPQ', 'BDPQ', 'MPQ', 'BDMQ', 'BHMQ', 'BMPQ', 'HMPQ']
METHODS = ['reflexion']
EXECUTION = 'GELLMO-LLAMA-REFLEXION-MAX3-SAMPLED-R20-20261023-v1'
ENVS = {n: '/ENVIRONMENT_ROOT/' + n + '/bin/python'
        for n in ('gellmo_mistral_gen', 'gellmo_inf', 'gellmo_eval_py36')}


def read(path):
    path = Path(path)
    with (gzip.open if path.suffix == '.gz' else open)(str(path), 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def destination(path):
    path = Path(path)
    assert path.is_absolute() and ROOT in path.resolve().parents and path == path.resolve(), str(path)
    return path


def save(path, value, replace=False):
    path = destination(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        assert read(path) == value, 'Immutable record differs: ' + str(path)
        return
    tmp = path.with_name('.' + path.name + '.tmp.' + str(os.getpid()))
    payload = (json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n').encode()
    with tmp.open('xb') as stream:
        if path.suffix == '.gz':
            with gzip.GzipFile(fileobj=stream, mode='wb', mtime=0) as compressed:
                compressed.write(payload)
        else:
            stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        if replace:
            os.replace(str(tmp), str(path))
        else:
            os.link(str(tmp), str(path))
    finally:
        if tmp.exists():
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
    for key in ('TMPDIR', 'HF_HOME', 'HF_HUB_CACHE', 'TRANSFORMERS_CACHE', 'XDG_CACHE_HOME',
                'TORCH_HOME', 'MPLCONFIGDIR', 'TRITON_CACHE_DIR', 'CUDA_CACHE_PATH', 'NUMBA_CACHE_DIR'):
        target = directory / key.lower()
        target.mkdir(parents=True, exist_ok=True)
        env[key] = str(target)
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
    frozen = read(ROOT / 'freeze_v4.json')
    for path, expected in frozen['sha256'].items():
        assert sha(path) == expected, path
    for item in read(LEGACY / 'evidence/model_stat_inventory.json'):
        st = Path(item['path']).stat()
        assert st.st_size == item['size'] and st.st_mtime_ns == item['mtime_ns'], item['path']
    assert read(ROOT.parent / 'recovery_20261027_v1/validation_gate.json')['status'] == 'PASS_NORMAL_EQUIVALENCE_AND_KNOWN_FAILURE_RECOVERY'


def records(unit):
    result = manifest(unit['task_id'])[unit['start']:unit['stop']]
    assert digest(result) == unit['records_sha256']
    return result


def batches(unit):
    rows = records(unit)
    for start in range(0, len(rows), 4):
        part = rows[start:start + 4]
        yield dict(batch_id=unit['unit_id'] + '_b{:02d}'.format(start // 4),
                   method='reflexion', task_id=unit['task_id'], records=part,
                   lanes=[dict(lane_id=r['input_uid'] + '_r{:02d}'.format(repeat), record=r, repeat=repeat)
                          for r in part for repeat in range(20)])


@lru_cache(None)
def property_spec(method, task):
    assert method == 'reflexion'
    return read(LEGACY / 'configs/properties_reflexion.json')['tasks'][task]


def terminal(method, lane, raw_smiles, raw_text, trace):
    record = lane['record']
    ident = dict(method_id='reflexion-max3-sampled-R20', model_id='gellmo_p6_llama',
                 task_id=record['task_id'], input_uid=record['input_uid'],
                 original_slot_id=lane['repeat'], execution=EXECUTION)
    return dict(ident, candidate_id='llama-reflexion-terminal-' + digest(ident),
                input_idx=record['input_idx'], source_smiles=record['source_smiles'],
                raw_smiles=raw_smiles, raw_text=raw_text, trace=trace)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj
