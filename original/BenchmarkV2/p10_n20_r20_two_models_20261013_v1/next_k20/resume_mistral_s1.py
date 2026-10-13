"""Resume immutable Mistral sampled-R20 s1 GENERATION into a new directory.

This does not change decoding or original RNG/batches, evaluate properties, submit
jobs, or write the old experiment. Run only after root GPU resource coordination.
"""
import argparse
import fcntl
import hashlib
import importlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OLD = REPO / 'BenchmarkV2/k20_remaining_three_v1'
STAGE = HERE / 'mistral_s1_resume'


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dependencies():
    sys.path.insert(0, str(OLD))
    common = importlib.import_module('common')
    worker = importlib.import_module('worker')
    # Redirect only the historical experiment's IO root. Model/prompt/evaluator
    # dependencies remain their exact immutable original paths. There is no
    # textual rewrite of model prompts, batch IDs, request keys, or seeds.
    for name in ['common', 'worker', 'replay', 'batched_generation', 'property_client']:
        mod = sys.modules[name]
        if getattr(mod, 'ROOT', None) == OLD:
            mod.ROOT = STAGE
    common.plan.cache_clear()
    return common, worker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--unit-index', type=int)
    ap.add_argument('--max-new-batches', type=int)
    ap.add_argument('--audit-only', action='store_true')
    args = ap.parse_args()
    assert STAGE.exists(), 'Run prepare_continuation.py first'
    for name in ['plan.json', 'units.json', 'inventory.json']:
        assert digest_file(STAGE/name) == digest_file(OLD/name)
    c,w = dependencies()
    c.verify()
    assert c.plan()['extensions']['s1']['main_id'] == 's1-B512-sampled-R20'
    if args.audit_only:
        assert c.ROOT == w.ROOT == STAGE
        assert sys.modules['replay'].ROOT == STAGE
        print(json.dumps(dict(status='PASS_CPU_ROOT_AND_SOURCE_CONTRACT_CHECK', old_root_read_only=str(OLD), new_write_root=str(STAGE))))
        return
    c.require_allocation()
    if args.max_new_batches is None:
        gate=json.loads((STAGE/'engineering_gate.json').read_text())
        assert gate['status']=='PASS' and gate['original_plan_sha256']==digest_file(OLD/'plan.json')
        assert gate['runner_sha256']==digest_file(__file__)
    unit = c.units()[args.unit_index]
    assert unit['method']=='s1' and unit['index']==args.unit_index
    expected = json.loads((HERE/'continuation_inventory.json').read_text())
    allowed = {u['index'] for u in expected['mistral_s1']['gpu_resume_units']}
    assert args.unit_index in allowed
    directory=STAGE/'runs'/unit['unit_id'];directory.mkdir(parents=True,exist_ok=True)
    lock=(directory/'.generation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    attempt=directory/('attempt_'+os.environ['SLURM_JOB_ID']+'_'+str(os.getpid()));attempt.mkdir(exist_ok=False)
    os.environ.update(c.isolate(attempt/'environment'))
    model=w.BatchedGeneration('s1',attempt)
    chosen=[];existing=[]
    for batch in c.batches(unit):
        original=OLD/'runs'/unit['unit_id']/batch['batch_id']
        output=directory/batch['batch_id']
        if (original/'generation.json.gz').exists():
            prev=c.read(original/'generation.json.gz')
            assert prev['batch']==batch and prev['plan_sha256']==digest_file(OLD/'plan.json')
            existing.append(dict(batch=batch['batch_id'],path=str(original/'generation.json.gz'),sha256=digest_file(original/'generation.json.gz')))
            continue
        output.mkdir(parents=True,exist_ok=True)
        if not (output/'generation.json.gz').exists():
            for src in sorted((original/'calls').glob('*.json.gz')):
                dest=output/'calls'/src.name;dest.parent.mkdir(parents=True,exist_ok=True)
                if dest.exists():assert digest_file(dest)==digest_file(src)
                else:shutil.copyfile(str(src),str(dest))
        chosen.append((batch,output))
    if args.max_new_batches is not None:chosen=chosen[:args.max_new_batches]
    class NoProperties:
        def evaluate(self,*a,**kw):raise AssertionError('s1 generation must not query properties')
    if any(not (folder/'generation.json.gz').exists() for batch,folder in chosen):model.start()
    results=[];start=time.time()
    for batch,folder in chosen:
        result=w.run_batch(batch,model,NoProperties(),folder)
        assert result['status']=='complete' and len(result['lanes'])==len(batch['lanes'])
        results.append(dict(batch=batch['batch_id'],slots=len(result['lanes']),path=str(folder/'generation.json.gz'),sha256=digest_file(folder/'generation.json.gz')))
        c.save(directory/'progress.json',dict(unit=unit['unit_id'],stage='generation_only',
               batches_done=len(results),batches_selected=len(chosen),terminal_slots=sum(r['slots'] for r in results),
               unix=time.time(),pending_terminal_property_evaluation=True),replace=True)
    c.save(attempt/'generation_receipt.json',dict(status='GENERATION_ONLY_COMPLETE_FOR_SELECTED_BATCHES',
           unit=unit,generated_selected=results,reused_original_batches=existing,
           pending_terminal_property_evaluation=True,original_runs_modified=False,
           seconds=time.time()-start,plan_sha256=digest_file(OLD/'plan.json')))
    print(json.dumps(dict(status='GENERATION_ONLY_COMPLETE_FOR_SELECTED_BATCHES',unit=unit['unit_id'],batches=len(results),terminal_slots=sum(r['slots'] for r in results))))


if __name__=='__main__':main()
