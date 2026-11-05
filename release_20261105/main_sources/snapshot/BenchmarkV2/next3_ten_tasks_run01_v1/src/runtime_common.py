"""Bounded Phase1D IO, environment and accounting; no model imports."""
import os
import sys
import time
import subprocess
from pathlib import Path
from contracts import ROOT, read, write, sha, verify

APPROVAL = 'BV2-NEXT3-TEN-TASKS-RUN01-v1'
ENVS = {name: '/ENVIRONMENT_ROOT/'+name+'/bin/python'
        for name in ('gellmo_mistral_gen','gellmo_inf','gellmo_eval_py36')}


def require_allocation(expected_unit=None, environ=None):
    from contracts import unit_spec
    env=os.environ if environ is None else environ
    for key in ['SLURM_JOB_ID','SLURM_ARRAY_JOB_ID','SLURM_ARRAY_TASK_ID']:
        if not env.get(key,'').isdigit():raise PermissionError('Approved real array allocation required')
    if env.get('SLURM_RESTART_COUNT','0')!='0':raise PermissionError('Restart forbidden')
    if env.get('BV2_NEXT3_APPROVAL_REF')!=APPROVAL:raise PermissionError('Wrong approval environment')
    if env.get('SLURM_CPUS_PER_TASK')!='8' or env.get('SLURM_GPUS_ON_NODE')!='1':
        raise PermissionError('Unit allocation requires 8 CPU and 1 GPU; CPU children inherit allocation')
    if (ROOT/'control/authorization_closed.json').exists():raise PermissionError('Stage closed')
    u=unit_spec(int(env['SLURM_ARRAY_TASK_ID']))
    if env.get('BV2_UNIT_ID')!=u['unit_id'] or (expected_unit is not None and expected_unit!=u['unit_id']):
        raise PermissionError('Array/unit identity mismatch')
    plan=read(ROOT/'control/batch_plan.json')['batches']
    bs=[b for b in plan if b['submission_key']==env.get('BV2_SUBMISSION_KEY') and u['array_index'] in b['indices']]
    if len(bs)!=1:raise PermissionError('Unapproved submission/index')
    b=bs[0];a=read(ROOT/'control'/('submission_'+b['batch_id']+'_attempt.json'))
    if a['approval_ref']!=APPROVAL or a['attempt']!=1 or a['indices']!=b['indices']:
        raise PermissionError('No matching once-only submission attempt')
    receipt=ROOT/'control'/('submission_'+b['batch_id']+'_receipt.json')
    if receipt.exists() and read(receipt).get('job_id')!=env['SLURM_ARRAY_JOB_ID']:
        raise PermissionError('Array parent differs from accepted receipt')
    return u


def isolated_env(directory, cpu=False):
    directory = Path(directory)
    if ROOT not in directory.resolve().parents or directory.absolute() != directory.resolve():
        raise ValueError('Environment directory escapes Phase1D')
    env = dict(os.environ)
    for name in ('PYTHONPATH','PYTHONHOME'):
        env.pop(name, None)
    for name, leaf in {'TMPDIR':'tmp','TMP':'tmp','TEMP':'tmp','HF_HOME':'cache/hf',
                       'HF_HUB_CACHE':'cache/hf/hub','TRANSFORMERS_CACHE':'cache/transformers',
                       'XDG_CACHE_HOME':'cache/xdg','TORCH_HOME':'cache/torch',
                       'MPLCONFIGDIR':'cache/matplotlib','TRITON_CACHE_DIR':'cache/triton',
                       'CUDA_CACHE_PATH':'cache/cuda','NUMBA_CACHE_DIR':'cache/numba'}.items():
        p = directory/leaf
        p.mkdir(parents=True, exist_ok=True)
        env[name] = str(p)
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',USE_TORCH='1',USE_TF='0',
               USE_FLAX='0',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8',
               MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    if cpu:
        env['CUDA_VISIBLE_DEVICES'] = ''
    return env


def checked_child(interpreter, script, arguments, directory, label, costs):
    cmd = [interpreter, '-B', str(ROOT/script)] + list(map(str, arguments))
    item = {'label':label,'argv':cmd,'started':True,'completed':False,
            'returncode':None,'wall_seconds':None,'cpu_gpu_visibility':''}
    costs.append(item)
    write(directory/(label+'_invocation.json'), item)
    start = time.perf_counter()
    with (directory/(label+'.stdout')).open('x') as out, (directory/(label+'.stderr')).open('x') as err:
        p = subprocess.run(cmd, stdout=out, stderr=err, env=isolated_env(directory,cpu=True), cwd=str(directory))
    item.update(returncode=p.returncode,completed=p.returncode==0,wall_seconds=time.perf_counter()-start)
    write(directory/(label+'_process.json'), item)
    if p.returncode:
        raise RuntimeError('Property/metric subprocess failed: '+label+' exit '+str(p.returncode))


def verify_code_inventory():
    for item in read(ROOT/'evidence/code_inventory_before_jobs.json'):
        verify(item['path'], item['sha256'])
    for item in read(ROOT/'evidence/model_weight_metadata_before.json')['files']:
        s=Path(item['path']).stat()
        if s.st_size!=item['size'] or s.st_mtime_ns!=item['mtime_ns']:
            raise ValueError('Previously verified model weight metadata changed')

    for item in read(ROOT/'evidence/property_asset_identity_before.json')['files']:
        st=Path(item['path']).stat()
        if st.st_size!=item['size'] or st.st_mtime_ns!=item['mtime_ns']:raise ValueError('Property asset identity changed')


def new_costs(method, cfg, n_inputs):
    return {'basis':'real fixed next3 full-task run01; returned tokens exclude internal beam work',
            'method_id':method,'planned_inputs':n_inputs,'generated_budget_per_input':cfg['generated_budget'],
            'visible_k':20,'planned_generate_calls':n_inputs*cfg['planned_calls_per_input'],
            'load_started':0,'load_completed':0,'generate_started':0,'generate_completed':0,
            'load_cuda_wall_seconds':None,'generate_cuda_wall_seconds':0.0,
            'prompt_tokens_unexpanded':0,'returned_tokens_through_first_eos':0,
            'calls':[],'property_processes':[],'property_events':[],
            'parent_selection_seconds':0.0,'generated_slots_actual':0,'inputs_complete':0,'evaluated_inputs_complete':0,
            'peak_allocated_bytes':None,'peak_reserved_bytes':None,'job_allocated_seconds':None,
            'worker_wall_seconds':None,'failure':None}
