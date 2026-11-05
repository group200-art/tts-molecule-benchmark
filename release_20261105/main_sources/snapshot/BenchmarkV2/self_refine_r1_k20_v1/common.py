"""Independent r1/K20 execution scope; immutable legacy scientific dependencies."""
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
LEGACY = REPO / 'BenchmarkV2/method_review_P5_v1/execution_v1'
sys.path.insert(0, str(LEGACY / 'src'))
from contracts import digest, manifest, TASKS, task_contract
from self_refine import prompt_for, task_text, parse_answer

METHOD = 'p5_self_refine_r1_k20_p6_v1'
EXECUTION = 'BV2-P5-SELFREFINE-R1-K20-v1'
ROLES = [('init', 0), ('feedback', 0), ('refine', 1), ('feedback', 1)]
ENVS = {n: '/ENVIRONMENT_ROOT/' + n + '/bin/python'
        for n in ('gellmo_mistral_gen', 'gellmo_inf', 'gellmo_eval_py36')}


def read(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(str(path), 'rt', encoding='utf-8') as f:
        return json.load(f)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value, replace=False):
    """Publish only complete files; never overwrite scientific records."""
    path = Path(path)
    assert ROOT in path.resolve().parents and path.absolute() == path.resolve(), str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.tmp.' + str(os.getpid()))
    payload = (json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n').encode()
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
    assert os.environ.get('SLURM_JOB_ID'), 'GPU/property work requires Slurm'
    assert os.environ.get('SR_K20_EXECUTION') == EXECUTION, 'Wrong execution scope'
    assert os.environ.get('SLURM_RESTART_COUNT', '0') == '0', 'Use explicit checkpoint resume'


def isolate(directory, cpu=False):
    directory = Path(directory)
    assert ROOT in directory.resolve().parents
    env = dict(os.environ)
    for key in ('PYTHONPATH', 'PYTHONHOME'):
        env.pop(key, None)
    for key, leaf in {'TMPDIR': 'tmp', 'HF_HOME': 'cache/hf', 'HF_HUB_CACHE': 'cache/hf/hub',
                      'XDG_CACHE_HOME': 'cache/xdg', 'TORCH_HOME': 'cache/torch',
                      'MPLCONFIGDIR': 'cache/matplotlib', 'TRITON_CACHE_DIR': 'cache/triton',
                      'CUDA_CACHE_PATH': 'cache/cuda', 'NUMBA_CACHE_DIR': 'cache/numba'}.items():
        p = directory / leaf
        p.mkdir(parents=True, exist_ok=True)
        env[key] = str(p)
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4' if cpu else '8',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
               USE_TORCH='1', USE_TF='0', USE_FLAX='0')
    if cpu:
        env['CUDA_VISIBLE_DEVICES'] = ''
    return env


def verify():
    for entry in read(ROOT / 'inventory.json'):
        assert sha(entry['path']) == entry['sha256'], entry['path']
    for name in ('model_metadata_before.json', 'property_metadata_before.json'):
        for entry in read(LEGACY / 'evidence' / name):
            st = Path(entry['path']).stat()
            assert st.st_size == entry['size'] and st.st_mtime_ns == entry['mtime_ns'], entry['path']


def plan():
    return read(ROOT / 'plan.json')


def records():
    return [record for task in TASKS for record in manifest(task)]


def batches(seed):
    assert 51004 <= seed <= 51020
    rows = records()
    for start in range(0, len(rows), 80):
        part = rows[start:start + 80]
        yield dict(batch_id='s{}_b{:03d}'.format(seed, start // 80), rep_seed=seed,
                   rows=part, row_identity_sha256=digest([(r['task_id'], r['input_uid']) for r in part]))


def batch_seed(batch, role, r):
    assert (role, r) in ROLES
    return int(digest(['SR-R1-K20-BATCH-v1', batch['batch_id'],
                       batch['row_identity_sha256'], role, r])[:15], 16) % 2147483647


def candidate(record, seed, answer, identity):
    parsed = parse_answer(answer['raw_text'])
    ident = dict(method_id=METHOD, model_id='gellmo_p6_mistral', task_id=record['task_id'],
                 input_uid=record['input_uid'], rep_seed=seed, original_slot_id=1,
                 scientific_identity_sha256=identity, raw_answer_sha256=digest(answer['raw_text']),
                 generation_seed=answer['seed'], batch_id=answer['batch_id'], batch_row=answer['batch_row'])
    return dict(ident, candidate_id='sr-r1-k20-' + digest(ident), input_idx=record['input_idx'],
                source_smiles=record['source_smiles'], raw_text=answer['raw_text'],
                raw_smiles=parsed['raw_smiles'], parse=parsed, call_index=answer['call_index'])


def audit_batch(result, batch, templates):
    assert result['batch_id'] == batch['batch_id'] and result['rep_seed'] == batch['rep_seed']
    assert result['plan_sha256'] == sha(ROOT / 'plan.json')
    assert result['status'] == 'complete' and len(result['lanes']) == len(batch['rows'])
    for index, (lane, record) in enumerate(zip(result['lanes'], batch['rows'])):
        assert lane['record'] == record
        assert len(lane['answers']) == len(lane['feedbacks']) == 2
        answers, feedbacks = [], []
        for role, r in ROLES:
            call = lane['feedbacks'][r] if role == 'feedback' else lane['answers'][r]
            prompt = prompt_for(role, task_text(record), answers, feedbacks, r, templates)
            assert call['prompt_digest'] == digest(prompt)
            assert call['seed'] == batch_seed(batch, role, r) and call['batch_row'] == index
            assert call['role'] == role and call['r'] == r and call['batch_id'] == batch['batch_id']
            (feedbacks if role == 'feedback' else answers).append(call)
        assert lane['terminal_parse'] == parse_answer(answers[1]['raw_text'])


def generated_candidates(seed, allow_partial=False):
    cfg = plan()
    templates = read(LEGACY / 'configs/prompts.json')
    items = []
    for batch in batches(seed):
        path = ROOT / 'generation' / str(seed) / 'batches' / (batch['batch_id'] + '.json.gz')
        if allow_partial and not path.exists():
            continue
        result = read(path)
        audit_batch(result, batch, templates)
        for lane in result['lanes']:
            c = candidate(lane['record'], seed, lane['answers'][1], cfg['scientific_identity_sha256'])
            c['trajectory_path'] = str(path)
            items.append(c)
    keys = [(c['task_id'], c['input_uid']) for c in items]
    assert len(keys) == len(set(keys))
    if not allow_partial:
        assert len(items) == 3905
        assert read(ROOT / 'generation' / str(seed) / 'complete.json')['status'] == 'complete'
    return items
