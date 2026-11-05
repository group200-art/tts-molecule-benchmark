"""Bounded second-model execution; immutable legacy scientific inputs."""
import ast
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
PHASE1A = REPO/'BenchmarkV2'
APPROVAL = 'BV2-P6-LLAMA-MAIN-TEN-TASKS-v1'
BUDGETS = (400,)


def unique_pairs(pairs):
    out = {}
    for k,v in pairs:
        if k in out: raise ValueError('Duplicate key: '+k)
        out[k]=v
    return out


def loads(text):
    def bad(value): raise ValueError('Nonfinite JSON: '+value)
    return json.loads(text,object_pairs_hook=unique_pairs,parse_constant=bad)


def read(path): return loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,allow_nan=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def verify(path,expected):
    if sha(path)!=expected: raise ValueError('SHA mismatch: '+str(path))


def destination(path):
    p=Path(os.path.abspath(str(path)))
    if ROOT not in p.parents or p.resolve()!=p: raise PermissionError('Output outside new stage: '+str(p))
    if os.path.lexists(str(p)): raise FileExistsError(str(p))
    return p


def write(path,value):
    p=destination(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f: json.dump(value,f,ensure_ascii=False,allow_nan=False,separators=(',',':')); f.write('\n')


def append(path,value):
    p=Path(os.path.abspath(str(path)))
    if ROOT not in p.parents or p.resolve()!=p: raise PermissionError(str(p))
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f: f.write(json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')


@lru_cache(None)
def evidence_path(relative):
    p=REPO/relative
    rows=[r for r in read(ROOT/'evidence/read_set.json') if r['path']==str(p)]
    if len(rows)!=1: raise ValueError('Unregistered scientific source: '+relative)
    verify(p,rows[0]['sha256']);return p


def pure_functions(relative,names,namespace=None):
    p=evidence_path(relative);tree=ast.parse(p.read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    if set(n.name for n in nodes)!=set(names):raise ValueError('Missing pure function')
    module=ast.parse('');module.body=nodes;ns={} if namespace is None else dict(namespace)
    exec(compile(ast.fix_missing_locations(module),str(p)+'::pure_only','exec'),ns);return ns


@lru_cache(None)
def kernel():
    p=evidence_path('BenchmarkV2/src/metrics_cpu.py');ns={}
    exec(compile(p.read_text(),str(p),'exec'),ns);return ns


@lru_cache(None)
def registry(): return {t['task_id']:t for t in read(evidence_path('BenchmarkV2/registries/task_registry.draft.json'))['tasks']}


def task_contract(task_id,supplied=None):
    t=registry()[task_id]
    if supplied is not None and supplied!=t['objectives']:raise ValueError('Task objectives changed')
    return t


@lru_cache(None)
def pilot_config():
    c=read(ROOT/'configs/full.json')
    assert c['approval_ref']==APPROVAL and not c['formal_freeze'] and c['model_id']=='gellmo_p6_llama'
    return c


def unit_spec(index):
    u=pilot_config()['units'][index]
    assert type(index) is int and u['array_index']==index and index>=0
    return u


def unit_for_id(unit_id):
    hits=[u for u in pilot_config()['units'] if u['unit_id']==unit_id]
    assert len(hits)==1;return hits[0]


@lru_cache(None)
def manifest(task_id,unit_id=None):
    t=task_contract(task_id);verify(t['manifest_path'],t['manifest_sha256'])
    rows=[loads(s) for s in Path(t['manifest_path']).read_text().splitlines()]
    assert len(rows)==t['actual_n'] and [r['input_idx'] for r in rows]==list(range(len(rows)))
    assert len({r['input_uid'] for r in rows})==len(rows)
    assert all(r['task_id']==task_id and r['task_key']==t['task_key'] for r in rows)
    if unit_id is not None:
        u=unit_for_id(unit_id);assert u['task_id']==task_id
        rows=rows[u['input_start']:u['input_stop_exclusive']]
        assert [r['input_uid'] for r in rows]==u['input_uids']
    return rows


@lru_cache(None)
def legacy_prompter():
    p=evidence_path('prompter.py')
    cls=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='Prompter')
    cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('__init__','format_prompt','generate_prompt','get_response')]
    constants={}
    for n in ast.parse(evidence_path('config.py').read_text()).body:
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('PROPERTY_FULL_NAMES','PROPERTY_IMPV_THRESHOLDS'):
            constants[n.targets[0].id]=ast.literal_eval(n.value)
    module=ast.parse('');module.body=[cls];ns=dict(constants,json=json,List=List)
    exec(compile(ast.fix_missing_locations(module),str(p)+'::pure_class','exec'),ns)
    evidence_path('templates/simple.json')
    return ns['Prompter']('simple','%%% Response:',str(REPO/'templates'))


def science(): return read(ROOT/'configs/p9_original_science.json')


def experiment_id(budget):
    assert budget==512
    return 'P9-B512-K1-v1'


def scientific_identity():
    return digest(dict(model='gellmo_p6_llama',method=pilot_config()['methods']['self_refine'],prompts=read(ROOT/'configs/p5_prompts.json')))
