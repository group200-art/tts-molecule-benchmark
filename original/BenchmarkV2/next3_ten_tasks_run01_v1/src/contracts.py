"""Phase 1B bounded, Python 3.6 compatible IO and read-only source contracts."""
import ast
import hashlib
import json
import os
import re
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
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
        raise ValueError('Output outside Phase1D or through symlink: ' + str(p))
    if os.path.lexists(str(p)):
        raise FileExistsError(str(p))
    return p


def write(path, value):
    p = destination(path)
    # No-clobber; reject symlinks before creating any directories or opening files.
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
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


def kernel():
    p = evidence_path('BenchmarkV2/src/metrics_cpu.py')
    ns = {}
    exec(compile(p.read_text(), str(p), 'exec'), ns)
    return ns


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


TASKS = ['BDP','BDQ','BPQ','DPQ','BDPQ','MPQ','BDMQ','BHMQ','BMPQ','HMPQ']
APPROVAL = 'BV2-NEXT3-TEN-TASKS-RUN01-v1'

def pilot_config():
    c=read(ROOT/'configs/full.json')
    if c['approval_ref']!=APPROVAL or c['formal_freeze'] is not False:
        raise PermissionError('Wrong bounded execution identity')
    original=read(ROOT/'evidence/phase1d/configs/pilot.json')
    if c['methods']!=original['methods'] or c['authorized_now']!=original['authorized_now']:
        raise ValueError('D scientific method/decoding changed')
    plan=read(ROOT/'evidence/attachments/BV2_NEXT3_TEN_TASKS_JOB_PLAN_v1.json')
    if c['hpc']!=plan['resource_proposal'] or [t['task_id'] for t in c['tasks']]!=TASKS:
        raise ValueError('Approved task/resource scope changed')
    if len(c['units'])!=60:
        raise ValueError('Exactly 60 approved units required')
    for u,p in zip(c['units'],plan['execution_units']):
        if any(u.get(k)!=v for k,v in p.items()):raise ValueError('Fixed array map changed')
    model=read(evidence_path('BenchmarkV2/metadata/gellmo_p6_mistral_model_provenance.json'))
    if c['model_id']!=model['model_id']:raise ValueError('Model identity changed')
    verify(Path(model['adapter_path'])/'adapter_config.json',model['adapter_config_json_sha256'])
    for name in ('config.json','generation_config.json','model.safetensors.index.json','tokenizer_config.json','special_tokens_map.json'):
        verify(Path(model['base_model']['path'])/name,model['base_model']['files'][name]['sha256'])
    return c

def unit_spec(index):
    if type(index) is not int or not 0<=index<60:raise ValueError('Unapproved array index')
    unit=pilot_config()['units'][index]
    if unit['array_index']!=index:raise ValueError('Index/unit mismatch')
    return unit

def unit_for_id(unit_id):
    units=[u for u in pilot_config()['units'] if u['unit_id']==unit_id]
    if len(units)!=1:raise ValueError('Unapproved unit ID')
    return units[0]

def manifest(task_id, unit_id=None):
    if task_id not in TASKS:raise ValueError('Task outside authorization')
    t=task_contract(task_id);verify(t['manifest_path'],t['manifest_sha256'])
    rows=[loads(line) for line in Path(t['manifest_path']).read_text().splitlines()]
    if len(rows)!=t['actual_n'] or [r['input_idx'] for r in rows]!=list(range(len(rows))):
        raise ValueError('Manifest count/order')
    if len({r['input_uid'] for r in rows})!=len(rows):raise ValueError('Duplicate manifest UID')
    for r in rows:
        if r['task_id']!=task_id or r['task_key']!=t['task_key']:raise ValueError('Manifest task identity')
    if unit_id is not None:
        u=unit_for_id(unit_id)
        if u['task_id']!=task_id:raise ValueError('Unit task differs')
        rows=rows[u['input_start']:u['input_stop_exclusive']]
        if len(rows)!=u['n_inputs'] or [r['input_uid'] for r in rows]!=u['input_uids']:
            raise ValueError('Unit UID/global-order mismatch')
    return rows


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
