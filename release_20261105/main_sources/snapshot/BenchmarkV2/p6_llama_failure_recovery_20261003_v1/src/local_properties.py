"""Original D/E CPU producer and run-local request cache, with recorded costs."""
from contracts import ROOT,read,write,sha,verify
from runtime_common import ENVS,checked_child
from property_bridge import requests_for


def request_batches(requests,limit=100):
    batches=[];batch=[];seen=set()
    for req in requests:
        raw=req['identity']['raw_smiles']
        if raw not in seen and len(seen)==limit:batches.append(batch);batch=[];seen=set()
        batch.append(req);seen.add(raw)
    if batch:batches.append(batch)
    return batches


class LocalProperties:
    def __init__(self,task,directory,costs,method):
        self.task,self.directory,self.costs,self.method=task,directory,costs,method
        filename='properties_'+method+'.json' if method in ('self_refine','fewshot','reflexion') else 'properties.json'
        self.spec=read(ROOT/'configs'/filename);self.contract=self.spec['tasks'][task]['cache_contract']
        self.cache={};self.events=0
        plan=read(ROOT/'configs/recovery.json')
        unit=next(u for u in plan['units'] if u['unit_id']==directory.name)
        reused=[]
        for entry in unit.get('completed_property_batches',[]):
            for item in entry['files']:verify(item['path'],item['sha256'])
            payload=read(entry['requests']);responses=read(entry['responses'])
            assert {r['request_id'] for r in responses}=={r['request_id'] for r in payload['requests']}
            for r in responses:
                assert r['cache_contract']==self.contract
                if r['request_id'] in self.cache:assert self.cache[r['request_id']]==r
                self.cache[r['request_id']]=r
            reused.append(entry)
        write(directory/'reused_property_batches.json',dict(batches=reused,requests=len(self.cache)))

    def bridge(self,candidates):
        requests,mapping=requests_for(candidates,self.contract)
        new=[r for r in requests if r['request_id'] not in self.cache]
        event=dict(index=self.events,candidate_slots=len(candidates),requested_unique=len(requests),cache_hits=len(requests)-len(new),new_requests=len(new),batches=[])
        self.costs['property_events'].append(event)
        for idx,batch in enumerate(request_batches(new)):
            d=self.directory/'properties'/('event{:04d}_batch{:03d}'.format(self.events,idx));d.mkdir(parents=True,exist_ok=False)
            inp=d/'requests.json';write(inp,dict(task_id=self.task,requests=batch,admet_columns=self.spec['tasks'][self.task]['admet_columns']))
            # Preserve each original method's actual DPQ path, including unused ADMET prediction.
            required=bool(self.spec['tasks'][self.task]['admet_columns']) or self.method in ('reflexion','self_refine','fewshot')
            folder='admet' if required else 'chemistry';script='admet_worker.py' if required else 'rdkit_recheck_worker.py'
            label='admet' if required else 'rdkit_recheck'
            checked_child(ENVS['gellmo_inf'],script,['--requests',inp,'--output',d/folder],d,label,self.costs['property_processes'])
            result=d/folder/('admet_result.json' if required else 'chemistry_result.json')
            checked_child(ENVS['gellmo_eval_py36'],'eval_worker.py',['--mode','produce','--input',inp,'--input-sha',sha(inp),'--admet',result,'--output',d/'standard'],d,'standard',self.costs['property_processes'])
            responses=read(d/'standard/responses_raw.json')
            assert len(responses)==len(batch) and {r['request_id'] for r in responses}=={r['request_id'] for r in batch}
            self.cache.update({r['request_id']:r for r in responses})
            event['batches'].append(dict(path=str(d),request_ids=[r['request_id'] for r in batch]))
        self.events+=1
        return dict(requests=requests,mapping=mapping,responses=[self.cache[r['request_id']] for r in requests],cache_contract=self.contract)
