"""One authorized generation unit; properties are absent from the process."""
import os
import sys
import time
import signal
import traceback
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from contracts import ROOT, METHOD, read, write, manifest, science, verify_inventory, scientific_identity, generation_directory
from runtime_common import require_allocation, isolated_env, slurm_environment, index_artifacts
from self_refine import run_input


def main():
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('--phase',choices=['E01','E02'],required=True)
    p.add_argument('--task'); p.add_argument('--index',type=int); args = p.parse_args()
    require_allocation(); verify_inventory()
    if args.phase == 'E01':
        directory = generation_directory('E01',task=args.task)
        seed=51001; task=args.task; rows=manifest(task)[:5]; spec=dict(task_id=task,start=0,end=5,rep_seed=seed)
        if os.environ.get('SLURM_ARRAY_JOB_ID'): raise ValueError('Pilot uses two non-array units')
    else:
        if read(ROOT/'results/e01_acceptance.json')['status'] != 'PASS': raise PermissionError('Pilot acceptance not PASS')
        if str(args.index) != os.environ.get('SLURM_ARRAY_TASK_ID'): raise ValueError('Array index mismatch')
        spec = read(ROOT/'configs/shards.json')[args.index]
        if spec['array_index'] != args.index: raise ValueError('Shard index mismatch')
        task=spec['task_id']; seed=spec['rep_seed']; rows=manifest(task)[spec['generate_start']:spec['end']]
        if [x['input_uid'] for x in rows] != spec['new_input_uids']: raise ValueError('Shard UID mismatch')
        directory = generation_directory('E02',index=args.index)
    directory.mkdir(parents=True,exist_ok=False)
    os.environ.update(isolated_env(directory)); os.chdir(str(directory))
    costs=dict(planned_inputs=len(rows),maximum_generate_calls=len(rows)*10,maximum_output_tokens=len(rows)*3250,
               load_started=0,load_completed=0,generate_started=0,generate_completed=0,
               load_cuda_wall_seconds=None,generate_cuda_wall_seconds=0.0,prompt_tokens_unexpanded=0,
               returned_tokens_through_first_eos=0,generated_slots_actual=0,inputs_complete=0,
               roles={r:dict(completed_calls=0,input_tokens=0,output_tokens=0,cuda_wall_seconds=0.0) for r in ['init','feedback','refine']},
               inference_tool_calls=0,failure=None)
    status=dict(phase=args.phase,spec=spec,method_id=METHOD,scientific_identity_sha256=scientific_identity(),
                status='started',slurm=slurm_environment(),started_at_unix=time.time())
    write(directory/'started.json',status); write(directory/'inputs.json',rows)
    started=time.perf_counter(); gen=None
    def terminate(signum,frame): raise RuntimeError('Termination signal {}; no retry'.format(signum))
    signal.signal(signal.SIGTERM,terminate)
    try:
        from generation_backend import RealGeneration
        gen=RealGeneration(METHOD,science(),costs,directory); gen.start()
        for record in rows:
            def emit(role,r,value):
                write(directory/'events'/('input{:04d}_{}_{}.json'.format(record['input_idx'],role,r)),value)
            result=run_input(record,seed,gen.generate,emit)
            write(directory/'trajectories'/('input{:04d}.json'.format(record['input_idx'])),result,compact=True)
            costs['inputs_complete']+=1
        if costs['generate_completed']!=len(rows)*10 or costs['generated_slots_actual']!=len(rows)*5:
            raise ValueError('Generation accounting mismatch')
        if costs['returned_tokens_through_first_eos']>len(rows)*3250: raise ValueError('Output token ceiling exceeded')
        if any(n=='rdkit' or n.startswith('rdkit.') or n=='admet_ai' or n=='properties' for n in sys.modules):
            raise RuntimeError('Property module imported into generation process')
        status['status']='complete'
    except BaseException as error:
        costs['failure']=repr(error); status['status']='failed'
        write(directory/'exception.json',dict(exception=repr(error),traceback=traceback.format_exc()))
        raise
    finally:
        if gen is not None and costs['load_started']:
            try: gen.memory()
            except Exception as error: costs['memory_measurement_failure']=repr(error)
        costs['worker_wall_seconds']=time.perf_counter()-started
        status['finished_at_unix']=time.time()
        write(directory/'costs.json',costs); write(directory/'status.json',status)
        index_artifacts(directory)


if __name__=='__main__': main()
