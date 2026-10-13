"""K20 identities and read-only access to the original task/metric definitions."""
import ast
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
APPROVAL = 'K20-ALIGNMENT-20261002-v1'


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key: ' + key)
        result[key] = value
    return result


def read(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=unique_pairs)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify(path, expected):
    if sha(path) != expected:
        raise ValueError('Source hash mismatch: ' + str(path))


def destination(path):
    path = Path(os.path.abspath(str(path)))
    if ROOT not in path.parents or path.resolve() != path:
        raise PermissionError('Output must stay inside the new experiment: ' + str(path))
    return path


def write(path, value):
    path = destination(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        handle.write('\n')


def append(path, value):
    path = destination(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')


@lru_cache(None)
def pilot_config():
    config = read(ROOT / 'configs/protocol.json')
    assert config['version'] == APPROVAL and config['visible_k'] == 20
    return config


def unit_spec(index):
    unit = pilot_config()['units'][index]
    assert type(index) is int and index >= 0 and unit['array_index'] == index
    return unit


def unit_for_id(uid):
    hits = [u for u in pilot_config()['units'] if u['unit_id'] == uid]
    if len(hits) != 1:
        raise ValueError('Unknown or ambiguous unit: ' + uid)
    return hits[0]


@lru_cache(None)
def registry():
    return {t['task_id']: t for t in read(REPO / 'BenchmarkV2/registries/task_registry.draft.json')['tasks']}


def task_contract(task_id, supplied=None):
    task = registry()[task_id]
    if supplied is not None and supplied != task['objectives']:
        raise ValueError('Task objectives changed')
    return task


@lru_cache(None)
def manifest(task_id, unit_id=None):
    task = task_contract(task_id)
    verify(task['manifest_path'], task['manifest_sha256'])
    rows = [json.loads(line) for line in Path(task['manifest_path']).read_text().splitlines()]
    assert len(rows) == task['actual_n']
    assert [r['input_idx'] for r in rows] == list(range(len(rows)))
    assert len({r['input_uid'] for r in rows}) == len(rows)
    if unit_id is not None:
        unit = unit_for_id(unit_id)
        assert unit['task_id'] == task_id
        rows = rows[unit['input_start']:unit['input_stop_exclusive']]
    return rows


@lru_cache(None)
def kernel():
    path = REPO / 'BenchmarkV2/src/metrics_cpu.py'
    namespace = {}
    exec(compile(path.read_text(), str(path), 'exec'), namespace)
    return namespace


@lru_cache(None)
def legacy_prompter():
    path = REPO / 'prompter.py'
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'Prompter')
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ('__init__', 'format_prompt', 'generate_prompt', 'get_response')]
    namespace = dict(json=json, List=List)
    for node in ast.parse((REPO / 'config.py').read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ('PROPERTY_FULL_NAMES', 'PROPERTY_IMPV_THRESHOLDS'):
            namespace[node.targets[0].id] = ast.literal_eval(node.value)
    module = ast.parse('')
    module.body = [cls]
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return namespace['Prompter']('simple', '%%% Response:', str(REPO / 'templates'))


def science():
    # Original parser and prompt definitions only. Effective stochastic decoding
    # comes exclusively from protocol.json and is logged for every physical call.
    return read(ROOT / 'configs/p9_original_science.json')


def experiment_id(budget):
    assert budget == 512
    return 's1-B512-sampled-R20'


def repeat_seed(method, task, uid, repeat):
    assert method in pilot_config()['methods'] and 0 <= repeat < 20
    return int(digest([APPROVAL, method, task, uid, repeat])[:8], 16)
