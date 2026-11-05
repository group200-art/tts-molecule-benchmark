"""Bounded allocation, isolated output/cache roots, and measured child processes."""
import os
import sys
import time
import subprocess
from pathlib import Path
from contracts import ROOT, APPROVAL, read, write

ENVS = {name:'/ENVIRONMENT_ROOT/'+name+'/bin/python'
        for name in ('gellmo_mistral_gen','gellmo_inf','gellmo_eval_py36')}


def require_allocation():
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_RESTART_COUNT','0') != '0':
        raise PermissionError('A fresh authorized Slurm allocation is required')
    if os.environ.get('P5_EXECUTION_ID') != APPROVAL:
        raise PermissionError('Wrong execution allocation')
    if (ROOT/'control/authorization_closed.json').exists():
        raise PermissionError('Execution authorization closed')


def isolated_env(directory, cpu=False):
    directory = Path(directory)
    if ROOT not in directory.resolve().parents or directory.absolute() != directory.resolve():
        raise ValueError('Runtime directory escapes work root')
    env = dict(os.environ)
    for name in ('PYTHONPATH','PYTHONHOME'): env.pop(name,None)
    for name,leaf in {'TMPDIR':'tmp','TMP':'tmp','TEMP':'tmp','HF_HOME':'cache/hf',
                      'HF_HUB_CACHE':'cache/hf/hub','TRANSFORMERS_CACHE':'cache/transformers',
                      'XDG_CACHE_HOME':'cache/xdg','TORCH_HOME':'cache/torch',
                      'MPLCONFIGDIR':'cache/matplotlib','TRITON_CACHE_DIR':'cache/triton',
                      'CUDA_CACHE_PATH':'cache/cuda','NUMBA_CACHE_DIR':'cache/numba'}.items():
        p = directory/leaf; p.mkdir(parents=True,exist_ok=True); env[name] = str(p)
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',USE_TORCH='1',USE_TF='0',
               USE_FLAX='0',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4' if cpu else '8',
               MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    if cpu: env['CUDA_VISIBLE_DEVICES'] = ''
    return env


def checked_child(interpreter, script, arguments, directory, label, costs):
    cmd = [interpreter,'-B',str(ROOT/script)]+list(map(str,arguments))
    item = dict(label=label,argv=cmd,started=True,completed=False,returncode=None,wall_seconds=None)
    costs.append(item); write(directory/(label+'_invocation.json'),item)
    start = time.perf_counter()
    with (directory/(label+'.stdout')).open('x') as out, (directory/(label+'.stderr')).open('x') as err:
        result = subprocess.run(cmd,stdout=out,stderr=err,env=isolated_env(directory,cpu=True),cwd=str(directory))
    item.update(returncode=result.returncode,completed=result.returncode==0,wall_seconds=time.perf_counter()-start)
    write(directory/(label+'_process.json'),item)
    if result.returncode: raise RuntimeError('Offline subprocess failed: '+label)


def slurm_environment():
    return {k:v for k,v in os.environ.items() if k.startswith('SLURM_') or k=='CUDA_VISIBLE_DEVICES'}


def index_artifacts(directory):
    from contracts import sha
    rows=[]
    for p in sorted(directory.rglob('*')):
        if not p.is_file() or any(part in ('cache','tmp') for part in p.relative_to(directory).parts):continue
        st=p.stat()
        rows.append(dict(path=str(p),size_bytes=st.st_size,mtime_ns=st.st_mtime_ns,sha256=sha(p)))
    write(directory/'raw_index.json',rows,compact=True)
