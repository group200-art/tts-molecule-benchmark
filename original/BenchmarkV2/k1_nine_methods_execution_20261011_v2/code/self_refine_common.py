"""R1 endpoint recovery: new output scope, immutable old scientific dependencies."""
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
LLAMA = REPO / 'BenchmarkV2/p6_llama_main_ten_tasks_v1'
RECOVERY = REPO / 'BenchmarkV2/p6_llama_failure_recovery_20261003_v1'
MISTRAL = REPO / 'BenchmarkV2/method_review_P5_v1/execution_v1'
NEXT3 = REPO / 'BenchmarkV2/next3_ten_tasks_run01_v1'
TASKS = ['BDP', 'BDQ', 'BPQ', 'DPQ', 'BDPQ', 'MPQ', 'BDMQ', 'BHMQ', 'BMPQ', 'HMPQ']
ENVS = {x: '/work/users/w/a/wanzhou/conda_envs/' + x + '/bin/python'
        for x in ('gellmo_inf', 'gellmo_eval_py36', 'gellmo_mistral_gen')}
sys.path.insert(0, str(LLAMA / 'src'))


def read(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(str(path), 'rt', encoding='utf-8') as handle:
        return json.load(handle)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def check_output(path):
    path = Path(path).absolute()
    assert ROOT in path.parents and path.resolve() == path, str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write(path, value, replace=False):
    path = check_output(path)
    if path.exists() and not replace:
        assert read(path) == value, 'Existing output differs: ' + str(path)
        return
    tmp = path.with_name('.' + path.name + '.tmp.' + str(os.getpid()))
    blob = (json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n').encode()
    with tmp.open('xb') as handle:
        if path.suffix == '.gz':
            with gzip.GzipFile(fileobj=handle, mode='wb', mtime=0) as compressed:
                compressed.write(blob)
        else:
            handle.write(blob)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(tmp), str(path))


def require_allocation():
    assert os.environ.get('SLURM_JOB_ID', '').isdigit(), 'Properties require CPU Slurm allocation'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU property job must hide GPUs'
    assert os.environ.get('SLURM_RESTART_COUNT', '0') == '0', 'Explicit resume only'


def isolated_env(directory):
    directory = check_output(Path(directory) / 'scope').parent
    env = dict(os.environ)
    for name in ('PYTHONPATH', 'PYTHONHOME'):
        env.pop(name, None)
    for name in ('TMPDIR', 'HF_HOME', 'HF_HUB_CACHE', 'XDG_CACHE_HOME', 'TORCH_HOME',
                 'MPLCONFIGDIR', 'TRITON_CACHE_DIR', 'CUDA_CACHE_PATH', 'NUMBA_CACHE_DIR'):
        target = directory / 'cache' / name.lower()
        target.mkdir(parents=True, exist_ok=True)
        env[name] = str(target)
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
               PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='8',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
               USE_TORCH='1', USE_TF='0', USE_FLAX='0')
    return env


def endpoint(record, candidate, model_id, provenance):
    from cache_metrics import evaluate_registered
    assert candidate['input_uid'] == record['input_uid']
    assert candidate['source_smiles'] == record['source_smiles']
    selected = dict(candidate, source_original_slot_id=candidate['original_slot_id'],
                    original_slot_id=0, visible_slot_id=0)
    metric = evaluate_registered(record['task_id'], record['input_uid'],
                                 record['source_properties'], [selected], 1)
    return dict(schema_version='BV2-K1-ENDPOINT-v2', model_id=model_id,
                method_id='self_refine_r1', task_id=record['task_id'],
                input_uid=record['input_uid'], input_idx=record['input_idx'],
                source_properties=record['source_properties'], selected_candidate=selected,
                selection_provenance=provenance, evaluation=metric)
