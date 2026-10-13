"""GoT private source binding inside the new Llama stage."""
import os
import sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from contracts import REPO,read,sha,digest,task_contract,kernel,append
from runtime_common import require_allocation,ENVS
INF_PY=ENVS['gellmo_inf']
EVAL_PY=ENVS['gellmo_eval_py36']
M1='P8-VALIDITY-GATE-INF-AUTHORITATIVE-v1'
M2='P8-inf-canonical-roundtrip-guard-v1'
CONTRACT=read(ROOT/'configs/design_contract.json')

def text_sha(value):
    import hashlib
    return hashlib.sha256(value.encode()).hexdigest()

def write(path,value,replace=False):
    import json
    p=Path(os.path.abspath(str(path)))
    assert ROOT in p.parents and p.resolve()==p
    p.parent.mkdir(parents=True,exist_ok=True)
    if not replace:
        with p.open('x') as f:json.dump(value,f,ensure_ascii=False,allow_nan=False,separators=(',',':'));f.write('\n')
    else:
        tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()))
        with tmp.open('x') as f:json.dump(value,f,ensure_ascii=False,allow_nan=False,separators=(',',':'));f.write('\n')
        os.replace(str(tmp),str(p))
