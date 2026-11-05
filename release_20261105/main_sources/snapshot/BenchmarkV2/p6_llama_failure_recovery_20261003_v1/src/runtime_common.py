import os
import subprocess
import time
from pathlib import Path
from contracts import ROOT,APPROVAL,read,write,verify,unit_spec

ENVS={name:'/ENVIRONMENT_ROOT/'+name+'/bin/python' for name in ('gellmo_mistral_gen','gellmo_inf','gellmo_eval_py36')}


def require_allocation(expected_unit=None):
    e=os.environ
    plan=read(ROOT/'configs/recovery.json')
    if e.get('BV2_LLAMA_RECOVERY')!=plan['recovery_id']:raise PermissionError('Wrong recovery authorization')
    for k in ('SLURM_JOB_ID','SLURM_ARRAY_JOB_ID','SLURM_ARRAY_TASK_ID'):
        if not e.get(k,'').isdigit():raise PermissionError('Actual Slurm array required')
    if e.get('SLURM_RESTART_COUNT','0')!='0':raise PermissionError('No automatic retries')
    if e.get('SLURM_CPUS_PER_TASK')!='8':raise PermissionError('8 CPU allocation required')
    if (ROOT/'control/authorization_closed.json').exists():raise PermissionError('Stage closed')
    u=unit_spec(int(e['SLURM_ARRAY_TASK_ID']))
    if expected_unit is not None and expected_unit!=u['unit_id']:raise PermissionError('Unit mapping mismatch')
    assert u['array_index'] in [x['index'] for x in plan['units']]
    batch='property_cpu' if u['method_id']=='sampling40_prefix20' else 'continuation_gpu'
    if batch=='continuation_gpu' and e.get('SLURM_GPUS_ON_NODE')!='1':raise PermissionError('One GPU required')
    if batch=='property_cpu' and e.get('CUDA_VISIBLE_DEVICES','')!='':raise PermissionError('Sampling recovery is CPU-only')
    intent=read(ROOT/'control'/('submission_'+batch+'_attempt01.json'))
    assert intent['execution_id']==plan['recovery_id'] and intent['attempt']==1 and u['array_index'] in intent['indices']
    receipt=ROOT/'control'/('submission_'+batch+'_receipt01.json')
    if receipt.exists():assert read(receipt)['job_id']==e['SLURM_ARRAY_JOB_ID']
    return u


def isolated_env(directory,cpu=False):
    directory=Path(directory)
    assert ROOT in directory.resolve().parents and directory.absolute()==directory.resolve()
    env=dict(os.environ)
    for k in ('PYTHONPATH','PYTHONHOME'):env.pop(k,None)
    for k in ('TMPDIR','TMP','TEMP','HF_HOME','HF_HUB_CACHE','TRANSFORMERS_CACHE','XDG_CACHE_HOME','TORCH_HOME','MPLCONFIGDIR','TRITON_CACHE_DIR','CUDA_CACHE_PATH','NUMBA_CACHE_DIR','JOBLIB_TEMP_FOLDER'):
        p=directory/'cache'/k.lower();p.mkdir(parents=True,exist_ok=True);env[k]=str(p)
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',USE_TORCH='1',USE_TF='0',USE_FLAX='0',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    if cpu:env['CUDA_VISIBLE_DEVICES']=''
    return env


def checked_child(interpreter,script,arguments,directory,label,costs):
    cmd=[interpreter,'-B',str(ROOT/script)]+list(map(str,arguments))
    item=dict(label=label,argv=cmd,started=True,completed=False,returncode=None,wall_seconds=None,cpu_gpu_visibility='')
    costs.append(item);write(directory/(label+'_invocation.json'),item);t=time.perf_counter()
    with (directory/(label+'.stdout')).open('x') as out,(directory/(label+'.stderr')).open('x') as err:
        p=subprocess.run(cmd,stdout=out,stderr=err,env=isolated_env(directory,cpu=True),cwd=str(directory))
    item.update(returncode=p.returncode,completed=p.returncode==0,wall_seconds=time.perf_counter()-t)
    write(directory/(label+'_process.json'),item)
    if p.returncode:raise RuntimeError('CPU child failed without retry: '+label+' '+str(p.returncode))


def verify_code_inventory():
    for row in read(ROOT/'evidence/code_inventory_before_jobs.json'):verify(row['path'],row['sha256'])
    for row in read(ROOT/'evidence/model_stat_inventory.json'):
        s=Path(row['path']).stat();assert (s.st_size,s.st_mtime_ns)==(row['size'],row['mtime_ns']),row['path']
    for row in read(ROOT/'evidence/property_asset_identity_before.json')['files']:
        s=Path(row['path']).stat();assert (s.st_size,s.st_mtime_ns)==(row['size'],row['mtime_ns']),row['path']
