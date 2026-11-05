"""P10 RUN01 bounded IO and immutable scientific/source identities (Python 3.6+)."""
import ast
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
RUN_ID = 'BV2-P10-EV-E01-E05-RUN01-v1'
METHOD_ID = 'p10_chen_vote_ev_native1_v1'
MODEL_ID = 'gellmo_p6_mistral'
TASKS = ['BDP','BDQ','BPQ','DPQ','BDPQ','MPQ','BDMQ','BHMQ','BMPQ','HMPQ']
BUDGETS = (1,20,100,400)
ENVS = {x:'/ENVIRONMENT_ROOT/'+x+'/bin/python'
        for x in ('gellmo_mistral_gen','gellmo_inf','gellmo_eval_py36')}


def unique_pairs(pairs):
    out = {}
    for k,v in pairs:
        if k in out: raise ValueError('Duplicate key: '+str(k))
        out[k] = v
    return out


def loads(text):
    def bad(v): raise ValueError('Nonstandard JSON constant: '+v)
    return json.loads(text,object_pairs_hook=unique_pairs,parse_constant=bad)


def read(path): return loads(Path(path).read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',', ':'),
                                    ensure_ascii=False,allow_nan=False).encode('utf-8')).hexdigest()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def verify(path,expected):
    if sha(path)!=expected: raise ValueError('SHA256 changed: '+str(path))


def destination(path,exclusive=True):
    p=Path(os.path.abspath(str(path)))
    if ROOT not in p.parents or p.resolve()!=p: raise ValueError('Output outside E or through symlink: '+str(p))
    if exclusive and os.path.lexists(str(p)): raise FileExistsError(str(p))
    return p


def write(path,value):
    p=destination(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,allow_nan=False,indent=2);f.write('\n')


def append_event(path,value):
    p=destination(path,exclusive=False);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a',encoding='utf-8') as f:
        f.write(json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',', ':'))+'\n')
        f.flush();os.fsync(f.fileno())


def task_contract(task_id,supplied=None):
    rows=read(ROOT/'configs/tasks.json')
    ts=[t for t in rows if t['task_id']==task_id]
    if len(ts)!=1: raise ValueError('Unapproved task')
    t=ts[0]
    if supplied is not None and supplied!=t['objectives']: raise ValueError('Objectives changed')
    return t


def manifest(task_id):
    t=task_contract(task_id);verify(t['manifest_path'],t['manifest_sha256'])
    rows=[loads(line) for line in Path(t['manifest_path']).read_text().splitlines()]
    if len(rows)!=t['actual_n'] or [r['input_idx'] for r in rows]!=list(range(len(rows))): raise ValueError('Manifest order/count')
    return rows


def unit_spec(phase,index):
    if phase not in ('pilot','full'): raise ValueError('Invalid phase')
    units=read(ROOT/'configs/plan.json')[phase]
    if type(index) is not int or not 0<=index<len(units): raise ValueError('Unapproved array index')
    u=units[index]
    if u['array_index']!=index: raise ValueError('Index mapping changed')
    return u


def seed_for(task_ord,input_idx,block):
    if type(task_ord) is not int or not 0<=task_ord<10 or type(input_idx) is not int or not 0<=input_idx<500 or type(block) is not int or not 0<=block<20:
        raise ValueError('Seed coordinate outside scope')
    return (10101+block)*100000+task_ord*1000+input_idx


def candidate_identity(record,slot):
    if type(slot) is not int or not 0<=slot<400: raise ValueError('Original slot outside pool')
    ident={'run_id':RUN_ID,'model_id':MODEL_ID,'method_id':METHOD_ID,'task_id':record['task_id'],
           'input_uid':record['input_uid'],'stage':'flat','node':'slot{}'.format(slot),'original_slot_id':slot}
    return dict(ident,candidate_id='candidate-v1-'+digest(ident))


def kernel():
    p=REPO/'BenchmarkV2/src/metrics_cpu.py'
    verify(p,read(ROOT/'configs/metric_contract.json')['kernel_sha256'])
    ns={};exec(compile(p.read_text(),str(p),'exec'),ns)
    return ns


def verify_code():
    for item in read(ROOT/'evidence/code_inventory_before_jobs.json'):
        verify(item['path'],item['sha256'])
    for item in read(ROOT/'evidence/protected_asset_metadata.json'):
        st=Path(item['path']).stat()
        if st.st_size!=item['size'] or st.st_mtime_ns!=item['mtime_ns']:
            raise ValueError('Protected asset metadata changed: '+item['path'])


def isolated_env(directory,cpu=False):
    d=destination(Path(directory)/'unused',exclusive=False).parent
    env=dict(os.environ)
    for k in ('PYTHONPATH','PYTHONHOME'): env.pop(k,None)
    for k,leaf in {'TMPDIR':'tmp','TMP':'tmp','TEMP':'tmp','HF_HOME':'cache/hf','HF_HUB_CACHE':'cache/hf/hub',
                   'TRANSFORMERS_CACHE':'cache/transformers','XDG_CACHE_HOME':'cache/xdg','TORCH_HOME':'cache/torch',
                   'MPLCONFIGDIR':'cache/matplotlib','TRITON_CACHE_DIR':'cache/triton','CUDA_CACHE_PATH':'cache/cuda',
                   'NUMBA_CACHE_DIR':'cache/numba'}.items():
        p=d/leaf;p.mkdir(parents=True,exist_ok=True);env[k]=str(p)
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
               HF_DATASETS_OFFLINE='1',USE_TORCH='1',USE_TF='0',USE_FLAX='0',TOKENIZERS_PARALLELISM='false',
               OMP_NUM_THREADS='8',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    if cpu: env['CUDA_VISIBLE_DEVICES']=''
    return env


def require_allocation(kind=None):
    e=os.environ
    if e.get('P10_RUN_ID')!=RUN_ID or not e.get('SLURM_JOB_ID','').isdigit() or e.get('SLURM_RESTART_COUNT','0')!='0':
        raise PermissionError('Single authorized Slurm allocation required')
    if (ROOT/'control/authorization_closed.json').exists(): raise PermissionError('Stage closed')
    if e.get('SLURM_JOB_ACCOUNT')!='SCHEDULER_ACCOUNT' or e.get('SLURM_CPUS_PER_TASK') not in ('8','4'):
        raise PermissionError('Allocation account/CPU mismatch')
    if 'login' in __import__('socket').gethostname().lower(): raise PermissionError('No science on login node')
    mode=e.get('P10_JOB_KIND')
    if kind and mode!=kind: raise PermissionError('Wrong job kind')
    if mode=='gpu':
        if e.get('SLURM_JOB_PARTITION')!='h100_all' or e.get('SLURM_JOB_NODELIST')!='GPU_HOST' or e.get('SLURM_CPUS_PER_TASK')!='8':
            raise PermissionError('Wrong GPU resource')
        if e.get('SLURM_GPUS_ON_NODE')!='1' or not e.get('SLURM_ARRAY_JOB_ID','').isdigit() or not e.get('SLURM_ARRAY_TASK_ID','').isdigit():
            raise PermissionError('Expected exactly one GPU per array worker')
        u=unit_spec(e['P10_PHASE'],int(e['SLURM_ARRAY_TASK_ID']))
        submission='gpu_'+e['P10_PHASE']
    elif mode in ('gate','cpu_full','audit'):
        if e.get('CUDA_VISIBLE_DEVICES')!='': raise PermissionError('CPU GPU visibility must be empty')
        submission=mode
        u=None
    else: raise PermissionError('Unapproved job kind')
    attempt=read(ROOT/'control'/(submission+'_submission_attempt.json'))
    if attempt['run_id']!=RUN_ID or attempt['attempt']!=1: raise PermissionError('No once-only submission')
    receipt=ROOT/'control'/(submission+'_submission_receipt.json')
    if receipt.exists():
        expected=read(receipt)['job_id']
        actual=e.get('SLURM_ARRAY_JOB_ID',e['SLURM_JOB_ID'])
        if actual!=expected: raise PermissionError('Job ID differs from receipt')
    return u
