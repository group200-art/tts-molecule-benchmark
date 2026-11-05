"""Self-Refine-local adaptation of the verified D/E IO and identity facade."""
import ast
import hashlib
import json
import os
import re
from pathlib import Path
from typing import List
from functools import lru_cache

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
PHASE1A = REPO / 'BenchmarkV2'


def unique_pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError('Duplicate JSON key: ' + key)
        out[key] = value
    return out


def loads(text):
    def bad_constant(value):
        raise ValueError('Nonstandard JSON constant: ' + value)
    return json.loads(text, object_pairs_hook=unique_pairs, parse_constant=bad_constant)


def read(path):
    return loads(Path(path).read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify(path, expected):
    if sha(path) != expected:
        raise ValueError('SHA256 mismatch: ' + str(path))


def destination(path):
    p = Path(os.path.abspath(str(path)))
    if ROOT not in p.parents or p.resolve() != p:
        raise ValueError('Output outside Self-Refine work root or through symlink: ' + str(p))
    if os.path.lexists(str(p)):
        raise FileExistsError(str(p))
    return p


def write(path, value, compact=False):
    p = destination(path)
    # No-clobber; reject symlinks before creating any directories or opening files.
    payload = json.dumps(value, indent=None if compact else 2, ensure_ascii=False, allow_nan=False) + '\n'
    p.parent.mkdir(parents=True, exist_ok=True)
    destination(p)
    with p.open('x', encoding='utf-8') as f:
        f.write(payload)


def evidence_path(relative):
    p = REPO / relative
    records = read(ROOT / 'evidence/read_set.json')
    matches = [r for r in records if r['path'] == str(p)]
    if len(matches) != 1:
        raise ValueError('Unregistered source: ' + relative)
    verify(p, matches[0]['sha256'])
    return p


def pure_functions(relative, names, namespace=None):
    """Extract only explicitly named pure functions; never import legacy modules."""
    p = evidence_path(relative)
    tree = ast.parse(p.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if set(n.name for n in nodes) != set(names):
        raise ValueError('Missing pure function')
    module = ast.parse('')
    module.body = nodes
    ns = {} if namespace is None else dict(namespace)
    exec(compile(ast.fix_missing_locations(module), str(p) + '::pure_only', 'exec'), ns)
    return ns


@lru_cache(maxsize=1)
def kernel():
    p = evidence_path('BenchmarkV2/src/metrics_cpu.py')
    ns = {}
    exec(compile(p.read_text(), str(p), 'exec'), ns)
    return ns


@lru_cache(maxsize=1)
def registry():
    data = read(evidence_path('BenchmarkV2/registries/task_registry.draft.json'))
    return unique_pairs([(t['task_id'], t) for t in data['tasks']])


def task_contract(task_id, supplied=None):
    tasks = registry()
    if task_id not in tasks:
        raise ValueError('Unknown task_id: ' + str(task_id))
    t = tasks[task_id]
    if supplied is not None:
        if not isinstance(supplied,list) or any(not isinstance(o,dict) or type(o.get('direction')) is not int or isinstance(o.get('supplementary_success_delta'),bool) for o in supplied):
            raise ValueError('Objectives require explicit integer directions and numeric non-boolean thresholds')
    if supplied is not None and supplied != t['objectives']:
        raise ValueError('Objectives/directions/delta thresholds must exactly match registry')
    return t


def legacy_prompter():
    # Restricted class AST avoids pandas/datasets and uses only the original template.
    p = evidence_path('prompter.py')
    cls = next(n for n in ast.parse(p.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'Prompter')
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ('__init__', 'format_prompt', 'generate_prompt', 'get_response')]
    config_tree = ast.parse(evidence_path('config.py').read_text())
    constants = {}
    for node in config_tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ('PROPERTY_FULL_NAMES', 'PROPERTY_IMPV_THRESHOLDS'):
            constants[node.targets[0].id] = ast.literal_eval(node.value)
    module = ast.parse(''); module.body = [cls]
    ns = dict(constants, json=json, List=List)
    exec(compile(ast.fix_missing_locations(module), str(p) + '::pure_class', 'exec'), ns)
    evidence_path('templates/simple.json')
    return ns['Prompter']('simple', '%%% Response:', str(REPO / 'templates'))


METHOD = 'p5_self_refine_neurips23_alg1_p6_v1'
APPROVAL = 'BV2-P5-SELFREFINE-E01-E03-v1'
TASKS = ['BDP','BDQ','BPQ','DPQ','BDPQ','MPQ','BDMQ','BHMQ','BMPQ','HMPQ']
SEEDS = [51001,51002,51003]


def science():
    c = read(ROOT/'configs/science.json')
    if c['method_id'] != METHOD or c['rep_seeds'] != SEEDS or c['refinements'] != 4:
        raise ValueError('Changed approved scientific configuration')
    return c


@lru_cache(maxsize=10)
def manifest(task_id):
    if task_id not in TASKS:
        raise ValueError('Unapproved task')
    t = task_contract(task_id)
    verify(t['manifest_path'], t['manifest_sha256'])
    rows = [loads(x) for x in Path(t['manifest_path']).read_text().splitlines()]
    if len(rows) != t['actual_n'] or [x['input_idx'] for x in rows] != list(range(len(rows))):
        raise ValueError('Manifest count/order mismatch')
    if len({x['input_uid'] for x in rows}) != len(rows):
        raise ValueError('Duplicate manifest UID')
    if any(x['task_id'] != task_id or x['task_key'] != t['task_key'] for x in rows):
        raise ValueError('Manifest task mismatch')
    return rows


def scientific_identity():
    return read(ROOT/'evidence/scientific_identity.json')['identity_sha256']


def verify_inventory():
    if (ROOT/'control/authorization_closed.json').exists():
        raise PermissionError('This execution scope is closed')
    auth = read(ROOT/'control/authorization.json')
    if auth['execution_id'] != APPROVAL or auth['user_explicitly_approved'] is not True:
        raise PermissionError('Missing explicit authorization')
    for item in read(ROOT/'evidence/code_inventory_before_jobs.json'):
        verify(item['path'], item['sha256'])
    for item in read(ROOT/'evidence/read_set.json'):
        verify(item['path'], item['sha256'])
    for name in ['model_metadata_before.json','property_metadata_before.json']:
        for item in read(ROOT/'evidence'/name):
            st = Path(item['path']).stat()
            if st.st_size != item['size'] or st.st_mtime_ns != item['mtime_ns']:
                raise ValueError('Protected dependency metadata changed: '+item['path'])


def generation_directory(phase, task=None, index=None):
    if phase == 'E01' and task in ('BDP','BHMQ'):
        return ROOT/'runs'/'E01'/task
    if phase == 'E02' and type(index) is int and 0 <= index < 120:
        return ROOT/'runs'/'E02'/('{:03d}'.format(index))
    raise ValueError('Unknown approved generation unit')


def evaluation_directory(phase, task, seed):
    if phase not in ('E01','E02','E03') or task not in TASKS or seed not in SEEDS:
        raise ValueError('Unknown evaluation unit')
    if phase == 'E01' and (task not in ('BDP','BHMQ') or seed != 51001):
        raise ValueError('Unknown pilot evaluation')
    return ROOT/'evaluation'/phase/str(seed)/task
