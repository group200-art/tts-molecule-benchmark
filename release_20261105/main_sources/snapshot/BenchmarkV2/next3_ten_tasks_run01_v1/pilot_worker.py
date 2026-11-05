"""Exactly one authorized Phase1D task x method, with local incremental properties."""
import sys, os, time, traceback
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from contracts import ROOT,read,write,sha,digest,pilot_config,manifest,task_contract,destination
from runtime_common import APPROVAL,ENVS,require_allocation,isolated_env,checked_child,new_costs,verify_code_inventory
from property_bridge import requests_for
from adapters import flat_prompt,tot_prompt,candidate,validate_candidates


def request_batches(requests,limit=100):
    # Each batch has <=100 unique raw strings, hence <=100 valid unique strings.
    batches=[];batch=[];seen=set()
    for req in requests:
        raw=req['identity']['raw_smiles']
        if raw not in seen and len(seen)==limit:
            batches.append(batch);batch=[];seen=set()
        batch.append(req);seen.add(raw)
    if batch:batches.append(batch)
    return batches

class LocalProperties:
    def __init__(self,task,directory,costs):
        self.task,self.directory,self.costs=task,directory,costs
        self.spec=read(ROOT/'configs/properties.json');self.contract=self.spec['tasks'][task]['cache_contract']
        self.cache={};self.events=0
    def bridge(self,candidates):
        requests,mapping=requests_for(candidates,self.contract);new=[r for r in requests if r['request_id'] not in self.cache]
        event=dict(index=self.events,candidate_slots=len(candidates),requested_unique=len(requests),cache_hits=len(requests)-len(new),new_requests=len(new),batches=[])
        self.costs['property_events'].append(event)
        for idx,batch in enumerate(request_batches(new)):
            d=self.directory/'properties'/('event{:02d}_batch{:02d}'.format(self.events,idx));d.mkdir(parents=True,exist_ok=False)
            inp=d/'requests.json';write(inp,dict(task_id=self.task,requests=batch,admet_columns=self.spec['tasks'][self.task]['admet_columns']))
            required=bool(self.spec['tasks'][self.task]['admet_columns'])
            folder='admet' if required else 'chemistry'
            script='admet_worker.py' if required else 'rdkit_recheck_worker.py'
            label='admet' if required else 'rdkit_recheck'
            checked_child(ENVS['gellmo_inf'],script,['--requests',inp,'--output',d/folder],d,label,self.costs['property_processes'])
            chemistry_result=d/folder/('admet_result.json' if required else 'chemistry_result.json')
            checked_child(ENVS['gellmo_eval_py36'],'eval_worker.py',['--mode','produce','--input',inp,'--input-sha',sha(inp),'--admet',chemistry_result,'--output',d/'standard'],d,'standard',self.costs['property_processes'])
            responses=read(d/'standard/responses_raw.json')
            if {r['request_id'] for r in responses}!={r['request_id'] for r in batch} or len(responses)!=len(batch):raise ValueError('Batch response mapping mismatch')
            self.cache.update({r['request_id']:r for r in responses})
            event['batches'].append(dict(path=str(d.relative_to(self.directory)),request_ids=[r['request_id'] for r in batch]))
        self.events+=1
        return dict(requests=requests,mapping=mapping,responses=[self.cache[r['request_id']] for r in requests],cache_contract=self.contract)

def execute(unit,directory,costs):
    task,method=unit['task_id'],unit['method_id']
    from generation_backend import RealGeneration
    cfg=pilot_config()['methods'][method];records=manifest(task,unit_id=unit['unit_id']);run_id=unit['run_id']
    write(directory/'inputs.json',records)
    gen=RealGeneration(method,cfg,costs,directory);gen.start()
    props=LocalProperties(task,directory,costs);groups={r['input_uid']:[] for r in records}
    for record in records:
        group=groups[record['input_uid']]
        if method!='p6_tot20':
            texts=gen.generate(flat_prompt(record),dict(record,stage='flat'))
            group.extend(candidate(run_id,method,record,'flat',i,'slot{}'.format(i),t) for i,t in enumerate(texts))
        else:
            for branch in range(5):
                text=gen.generate(tot_prompt(record,branch),dict(record,stage='depth1',branch=branch),601+record['input_idx']*100+branch)[0]
                group.append(candidate(run_id,method,record,'depth1',branch,'b{}'.format(branch),text))
        write(directory/'candidates'/('input{}_{}.json'.format(record['input_idx'],'depth1' if method=='p6_tot20' else 'flat')),group)
    if method=='p6_tot20':
        d1=[c for r in records for c in groups[r['input_uid']]];bridge=props.bridge(d1)
        payload=[]
        for record in records:
            group=groups[record['input_uid']];requests,mapping=requests_for(group,props.contract);rids={r['request_id'] for r in requests}
            b=dict(bridge,requests=requests,mapping=mapping,responses=[r for r in bridge['responses'] if r['request_id'] in rids])
            payload.append({'record':record,'depth1':list(group),'bridge':b})
        inp=directory/'parent_input.json';write(inp,payload)
        checked_child(ENVS['gellmo_eval_py36'],'eval_worker.py',['--mode','parents','--input',inp,'--input-sha',sha(inp),'--output',directory/'parents.json'],directory,'parents',costs['property_processes'])
        parents=read(directory/'parents.json');costs['parent_selection_seconds']=parents['selection_wall_seconds']
        for record,item in zip(records,parents['parents']):
            if item['record']!=record or item['depth1']!=groups[record['input_uid']] or item['gate']['depth1_digest']!=digest(item['depth1']) or item['gate']['property_bridge_digest']!=digest(item['bridge']) or item['gate']['parent_digest']!=digest(item['gate']['ranked']):
                raise ValueError('ToT dependency digest/input mismatch')
            if len(item['plans'])!=15:raise ValueError('ToT children count')
            group=groups[record['input_uid']]
            for plan in item['plans']:
                text=gen.generate(plan['prompt'],dict(plan,input_uid=record['input_uid'],input_idx=record['input_idx']),plan['seed'])[0]
                group.append(candidate(run_id,method,record,'depth2',plan['original_slot_id'],plan['node'],text))
            write(directory/'candidates'/('input{}_complete.json'.format(record['input_idx'])),group)
    candidates=[c for r in records for c in groups[r['input_uid']]]
    validate_candidates(candidates,records,method,run_id);costs['inputs_complete']=len(records)
    write(directory/'candidates.json',candidates)
    bridge=props.bridge([c for c in candidates if c['original_slot_id']<20])
    bundle=dict(bridge,unit_id=unit['unit_id'],run_id=run_id,model_id='gellmo_p6_mistral',task_id=task,method_id=method,
                inputs=records,objectives=task_contract(task)['objectives'],candidates=candidates)
    inp=directory/'bundle.json';write(inp,bundle)
    checked_child(ENVS['gellmo_eval_py36'],'eval_worker.py',['--mode','metrics','--input',inp,'--input-sha',sha(inp),
                   '--output',directory/'metrics.json'],directory,'metrics',costs['property_processes'])
    costs['evaluated_inputs_complete']=len(records)
    if costs['generated_slots_actual']!=len(records)*cfg['generated_budget']:raise ValueError('Molecular slot count mismatch')
    if costs['returned_tokens_through_first_eos']>len(records)*cfg['generated_budget']*cfg['decoding']['max_new_tokens']:raise ValueError('Returned token cap exceeded')
    if costs['generate_started']!=costs['planned_generate_calls'] or costs['generate_completed']!=costs['planned_generate_calls']:
        raise ValueError('Physical generation calls do not match method budget')
    gen.memory()


def main():
    from contracts import unit_spec
    index=os.environ.get('SLURM_ARRAY_TASK_ID','')
    if not index.isdigit():raise PermissionError('Array index required')
    unit=unit_spec(int(index));os.environ['BV2_UNIT_ID']=unit['unit_id']
    if read(ROOT/'control/authorization.json')['user_explicitly_approved'] is not True:
        raise PermissionError('Missing recorded user approval')
    require_allocation(unit['unit_id']);verify_code_inventory()
    task,method=unit['task_id'],unit['method_id'];cfg=pilot_config()['methods'][method]
    directory=destination(ROOT/'runs'/unit['unit_id']/'run01');directory.mkdir(parents=True,exist_ok=False)
    os.environ.update(isolated_env(directory));os.chdir(str(directory))
    costs=new_costs(method,cfg,unit['n_inputs']);start=time.perf_counter()
    status=dict(task_id=task,method_id=method,unit_id=unit['unit_id'],array_index=unit['array_index'],run_id=unit['run_id'],status='started',approval_ref=APPROVAL,
                slurm={k:v for k,v in os.environ.items() if k.startswith('SLURM_') or k=='CUDA_VISIBLE_DEVICES'})
    write(directory/'started.json',status)
    try:
        execute(unit,directory,costs);status['status']='pass'
    except BaseException as error:
        status['status']='failed';costs['failure']=repr(error)
        write(directory/'exception.json',dict(exception=repr(error),traceback=traceback.format_exc()));raise
    finally:
        costs['worker_wall_seconds']=time.perf_counter()-start
        if 'torch' in sys.modules and costs['load_started']:
            torch=sys.modules['torch']
            costs['peak_allocated_bytes']=torch.cuda.max_memory_allocated(0)
            costs['peak_reserved_bytes']=torch.cuda.max_memory_reserved(0)
        write(directory/'costs.json',costs);write(directory/'status.json',status)

if __name__=='__main__':main()
