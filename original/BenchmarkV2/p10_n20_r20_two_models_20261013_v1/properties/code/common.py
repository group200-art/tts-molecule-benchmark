"""Scope-safe utilities for property evaluation of fixed N20/R20 winners."""
import gzip,hashlib,json,os,sys
from pathlib import Path
sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parents[2]
LLAMA=ROOT/'BenchmarkV2/p6_llama_main_ten_tasks_v1'
RECOVERY=ROOT/'BenchmarkV2/p6_llama_failure_recovery_20261003_v1'
P10=ROOT/'BenchmarkV2/method_review_P10_v1/execution_vote_ev_v1'
TASKS=['BDP','BDQ','BPQ','DPQ','BDPQ','MPQ','BDMQ','BHMQ','BMPQ','HMPQ']
ENVS={name:'/work/users/w/a/wanzhou/conda_envs/'+name+'/bin/python' for name in ['gellmo_inf','gellmo_eval_py36','gellmo_mistral_gen']}

def read(path):
    p=Path(path)
    with (gzip.open(str(p),'rt') if p.suffix=='.gz' else p.open()) as f:return json.load(f)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()).hexdigest()

def write(path,obj):
    p=Path(path).absolute();assert OUT in p.parents and p.resolve()==p
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():assert read(p)==obj,'Immutable existing output differs: '+str(p);return
    tmp=p.with_name('.'+p.name+'.tmp.'+str(os.getpid()))
    blob=(json.dumps(obj,ensure_ascii=False,allow_nan=False)+'\n').encode()
    with tmp.open('xb') as f:
        if p.suffix=='.gz':
            with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(blob)
        else:f.write(blob)
    os.rename(str(tmp),str(p))

def require_allocation():
    assert os.environ.get('SLURM_JOB_ID','').isdigit()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''

def isolated(directory):
    d=Path(directory);assert OUT in d.parents
    env=dict(os.environ)
    for n in ['PYTHONPATH','PYTHONHOME']:env.pop(n,None)
    for n in ['TMPDIR','HF_HOME','HF_HUB_CACHE','XDG_CACHE_HOME','TORCH_HOME','MPLCONFIGDIR','TRITON_CACHE_DIR','CUDA_CACHE_PATH','NUMBA_CACHE_DIR']:
        p=d/'runtime_cache'/n.lower();p.mkdir(parents=True,exist_ok=True);env[n]=str(p)
    env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',USE_TORCH='1',USE_TF='0',USE_FLAX='0')
    return env

def known_failure(response):
    e=response.get('property_error')
    if e is not None:return e.get('policy_id')=='rdkit_remove_hs_property_error_isolation_v1' and e.get('category')=='rdkit_kekulization_after_initial_parse' and e.get('initial_parse_valid') is True
    v={x['environment']:x for x in response['canonical_representations']}
    return bool(response.get('environment_validity_disagreement') and v['gellmo_inf']['valid'] and not v['gellmo_eval_py36']['valid'] and response.get('missing_reason')=='invalid_in_inf_or_eval')

def verify_dependencies():
    for e in read(OUT/'config/dependencies.json'):assert sha(e['path'])==e['sha256'],e['path']
    config=read(OUT/'config/properties.json')
    for path,h in config['identity_files'].items():assert sha(path)==h,path
    return config
