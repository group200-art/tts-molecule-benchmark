"""One fixed Llama unit, once. All model outputs and native selection are retained."""
import os
import signal
import sys
import time
import traceback
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from contracts import ROOT,APPROVAL,read,write,sha,digest,pilot_config,manifest,destination,task_contract
from runtime_common import ENVS,require_allocation,isolated_env,checked_child,verify_code_inventory
from local_properties import LocalProperties


def molecular_candidate(u,r,slot,text,smiles,**extra):
    identity=dict(run_id=APPROVAL,model_id='gellmo_p6_llama',method_id=u['method_id'],task_id=r['task_id'],input_uid=r['input_uid'],original_slot_id=slot)
    return dict(identity,candidate_id='llama-candidate-v1-'+digest(identity),input_idx=r['input_idx'],source_smiles=r['source_smiles'],raw_text=text,raw_smiles=smiles,**extra)


def native_metrics(u,records,d,costs,candidates=None,bridge=None,filled_groups=None):
    data=dict(unit_id=u['unit_id'],records=records,model_id='gellmo_p6_llama')
    if bridge is not None:data.update(candidates=candidates,bridge=bridge)
    else:data['filled_groups']=filled_groups
    inp=d/'native_bundle.json';write(inp,data)
    checked_child(ENVS['gellmo_eval_py36'],'eval_worker.py',['--mode','native','--input',inp,'--input-sha',sha(inp),'--output',d/'metrics.json'],d,'native_metrics',costs['property_processes'])
    result=read(d/'metrics.json');assert [r['input_uid'] for r in result['inputs']]==u['input_uids']
    costs['evaluated_inputs_complete']=len(records)


def run_flat_or_tot(u,records,gen,props,d,costs):
    from adapters import flat_prompt,tot_prompt,candidate,validate_candidates
    from property_bridge import requests_for
    method=u['method_id'];groups={r['input_uid']:[] for r in records}
    for r in records:
        group=groups[r['input_uid']]
        if method!='p6_tot20':
            outputs=gen.generate(flat_prompt(r),dict(r,stage='flat'))
            group.extend(candidate(APPROVAL,method,r,'flat',i,'slot'+str(i),o['raw_text']) for i,o in enumerate(outputs))
        else:
            for b in range(5):
                o=gen.generate(tot_prompt(r,b),dict(r,stage='depth1',branch=b),601+r['input_idx']*100+b)[0]
                group.append(candidate(APPROVAL,method,r,'depth1',b,'b'+str(b),o['raw_text']))
        write(d/'generated'/('{:04d}_initial.json'.format(r['input_idx'])),group)
    if method=='p6_tot20':
        d1=[c for r in records for c in groups[r['input_uid']]];bridge=props.bridge(d1);payload=[]
        for r in records:
            group=groups[r['input_uid']];requests,mapping=requests_for(group,props.contract);ids={q['request_id'] for q in requests}
            payload.append(dict(record=r,depth1=list(group),bridge=dict(bridge,requests=requests,mapping=mapping,responses=[x for x in bridge['responses'] if x['request_id'] in ids])))
        inp=d/'parent_input.json';write(inp,payload)
        checked_child(ENVS['gellmo_eval_py36'],'eval_worker.py',['--mode','parents','--input',inp,'--input-sha',sha(inp),'--output',d/'parents.json'],d,'parents',costs['property_processes'])
        parents=read(d/'parents.json');assert len(parents['parents'])==len(records)
        for r,item in zip(records,parents['parents']):
            assert item['record']==r and item['depth1']==groups[r['input_uid']]
            assert item['gate']['depth1_digest']==digest(item['depth1']) and item['gate']['property_bridge_digest']==digest(item['bridge'])
            assert len(item['plans'])==15
            for plan in item['plans']:
                o=gen.generate(plan['prompt'],dict(plan,input_uid=r['input_uid'],input_idx=r['input_idx']),plan['seed'])[0]
                groups[r['input_uid']].append(candidate(APPROVAL,method,r,'depth2',plan['original_slot_id'],plan['node'],o['raw_text']))
            write(d/'generated'/('{:04d}_complete.json'.format(r['input_idx'])),groups[r['input_uid']])
    all_candidates=[c for r in records for c in groups[r['input_uid']]]
    validate_candidates(all_candidates,records,method,APPROVAL)
    write(d/'all_candidates.json',all_candidates)
    visible=[c for c in all_candidates if c['original_slot_id']<20]
    costs['inputs_complete']=len(records)
    native_metrics(u,records,d,costs,visible,props.bridge(visible))


def run_reflexion(u,records,gen,props,d,costs):
    from reflexion import run_input
    templates=read(ROOT/'configs/reflexion_prompts.json');filled=[]
    for r in records:
        folder=d/'trajectories'/('{:04d}'.format(r['input_idx']));folder.mkdir(parents=True,exist_ok=False)
        def emit(trial,name,value):write(folder/('trial{:02d}_{}.json'.format(trial,name)),value)
        def call(role,prompt,context):return dict(gen.generate(prompt,context)[0],role=role)
        def evaluate(c):
            bridge=props.bridge([c]);trial=c['trial_index'];inp=folder/('trial{:02d}_eval_input.json'.format(trial));out=folder/('trial{:02d}_eval_result.json'.format(trial))
            write(inp,dict(record=r,candidate=c,bridge=bridge))
            checked_child(ENVS['gellmo_eval_py36'],'eval_worker.py',['--mode','evaluate_trial','--input',inp,'--input-sha',sha(inp),'--output',out],folder,'evaluate_trial'+str(trial),costs['property_processes'])
            return read(out)
        result=run_input(r,APPROVAL,templates,call,evaluate,emit)
        write(folder/'complete.json',result)
        filled.append(dict(record=r,candidates=[result['trials'][-1]['evaluation']['filled_candidate']]))
        costs['inputs_complete']+=1
    native_metrics(u,records,d,costs,filled_groups=filled)


def run_terminal(u,records,gen,props,d,costs):
    method=u['method_id'];candidates=[];ev_data=[]
    if method=='ev':
        plans={r['input_uid']:r for r in read(ROOT/'configs/p10_prompts.json')}
    if method=='fewshot':
        plans={r['record']['input_uid']:r for r in read(ROOT/('configs/pplus_prompts_'+u['task_id']+'.json'))}
    for r in records:
        folder=d/'trajectories'/('{:04d}'.format(r['input_idx']));folder.mkdir(parents=True,exist_ok=False)
        if method=='self_refine':
            from self_refine import run_input,candidate
            def call(role,prompt,context,seed):
                spec=dict(gen.cfg['decoding'],max_new_tokens=gen.cfg['role_caps'][role])
                return gen.generate(prompt,context,seed,decoding=spec)[0]
            def emit(role,round_index,value):write(folder/(role+str(round_index)+'.json'),value)
            trajectory=run_input(r,51001,call,emit)
            c=candidate(r,51001,4,trajectory['answers'][4],trajectory['scientific_identity_sha256'])
        elif method=='s1':
            from p9_adapters import build_prompt,parse_final
            from s1bf import run_one_budget
            prompt=build_prompt(r);ids=gen.tok.encode(prompt,add_special_tokens=True,truncation=False)
            trajectory=run_one_budget(gen,ids,512,dict(task_id=r['task_id'],input_uid=r['input_uid'],input_idx=r['input_idx']))
            parsed=parse_final(trajectory['final_text']);c=molecular_candidate(u,r,0,trajectory['final_text'],parsed['raw_smiles'],parser=parsed)
        elif method=='fewshot':
            from pplus_parser import parse_smiles
            plan=plans[r['input_uid']];assert plan['record']==r and plan['shots']==4
            outputs=gen.generate(plan['prompt'],dict(task_id=r['task_id'],input_uid=r['input_uid'],input_idx=r['input_idx'],shots=4),plan['generation_seed'])
            assert len(outputs)==4
            parsed=parse_smiles(outputs[0]['raw_text'])
            c=molecular_candidate(u,r,0,outputs[0]['raw_text'],parsed[0],parser=list(parsed),beam_rank=0)
            trajectory=dict(plan=plan,outputs=outputs,native_beam_rank=0)
        elif method=='ev':
            from p10_parser import extract_candidate
            plan=plans[r['input_uid']]
            assert all(plan[k]==r[k] for k in ('input_uid','input_idx','task_id')) and plan['task_ord']==u['task_ord']
            proposals=[]
            for b in range(20):
                seed=(10101+b)*100000+u['task_ord']*1000+r['input_idx']
                outputs=gen.generate(plan['prompt'],dict(task_id=r['task_id'],input_uid=r['input_uid'],input_idx=r['input_idx'],block=b),seed)
                assert len(outputs)==20
                block=[]
                for s,o in enumerate(outputs):
                    slot=20*b+s;raw=o['raw_text'].strip();parsed=extract_candidate(raw)
                    row=molecular_candidate(u,r,slot,raw,parsed['parsed_candidate'],global_slot=slot,block=b,block_slot=s,seed=seed,raw_response=raw,call_path=o['call_path'],**parsed)
                    proposals.append(row);block.append(row)
                write(folder/('block{:02d}.json'.format(b)),block)
            assert len(proposals)==400
            ev_data.append(dict(record=r,proposals=proposals))
            trajectory=dict(record=r,physical_calls=20,proposals=400)
            c=None
        else:raise ValueError(method)
        write(folder/'complete.json',trajectory)
        if c is not None:candidates.append(c)
        costs['inputs_complete']+=1
    if method=='ev':
        inp=d/'vote_input.json';out=d/'vote_result.json';write(inp,ev_data)
        checked_child(ENVS['gellmo_inf'],'vote_worker.py',['--input',inp,'--input-sha',sha(inp),'--output',out],d,'exact_molecular_vote',costs['property_processes'])
        votes=read(out);assert [x['record'] for x in votes['inputs']]==records
        candidates=[x['native'] for x in votes['inputs']]
    assert len(candidates)==len(records)
    write(d/'native_candidates.json',candidates)
    native_metrics(u,records,d,costs,candidates,props.bridge(candidates))


def run_got(u,records,d,costs):
    sys.path.insert(0,str(ROOT/'code'))
    from recovery_backend import Backend
    from recovery_properties import PropertyBridge
    from graph_adapter import MolecularGraph
    from common import write as replaceable_write
    from scorer import score_pool
    backend=Backend(d,u['got_logical_unit']);props=PropertyBridge(u['task_id'],d);groups=[]
    try:
        backend.start();props.start();backend.begin_G(12)
        for r in records:
            path=d/'graphs'/('{:04d}.json'.format(r['input_idx']))
            def persist(event,graph):replaceable_write(path,dict(graph.snapshot(event),model_id='gellmo_p6_llama',run_id=APPROVAL),replace=True)
            graph=MolecularGraph(r,12,backend,props,persist);saved=graph.run()
            assert len(saved['nodes'])==28 and len(saved['top20'])==20 and saved['event']=='complete'
            assert score_pool(saved['nodes'],saved['properties'],r['source_properties'],task_contract(r['task_id'])['objectives'])==saved['scores']['full']
            nodes={n['node_id']:n for n in saved['nodes']};cs=[]
            for rank,nid in enumerate(saved['top20']):
                p=saved['properties'][nid];n=nodes[nid]
                cs.append(dict(candidate_id=nid,original_slot_id=rank,created_order=n['created_order'],raw_smiles=p['request_smiles'],
                    rdkit_valid=p['rdkit_valid'],properties=p['properties'],property_row_exists=p['property_row_exists'],model_id='gellmo_p6_llama',input_uid=r['input_uid'],
                    property_error=p.get('property_error'),environment_validity_disagreement=p.get('environment_validity_disagreement',False)))
            groups.append(dict(record=r,candidates=cs));costs['inputs_complete']+=1
        assert backend.cost['calls_started']==backend.cost['calls_completed']==28*len(records)
        assert backend.cost['reused_calls']==len(backend.saved)
        native_metrics(u,records,d,costs,filled_groups=groups)
    finally:
        errors=[]
        for obj in (props,backend):
            try:obj.close()
            except BaseException as err:errors.append(repr(err))
        c=backend.cost;g=c['by_G'].get('12',{})
        costs.update(load_started=c['model_load_started'],load_completed=c['model_load_completed'],load_cuda_wall_seconds=c['model_load_seconds'],
            generate_started=c['calls_started'],generate_completed=c['calls_completed'],generate_cuda_wall_seconds=g.get('generate_seconds',0),
            physical_return_sequences=c['calls_completed'],returned_tokens_through_first_eos=g.get('output_tokens_through_eos',0),prompt_tokens_unexpanded=g.get('prompt_tokens',0),close_errors=errors)
        costs.update(reused_generation_calls=c['reused_calls'],new_generation_calls=c['new_calls'],
                     reused_generation_cuda_seconds=c['reused_generate_seconds'],new_model_loads=c['model_load_completed'])
        if errors:raise RuntimeError('GoT resource close error: '+repr(errors))


def main():
    u=require_allocation();verify_code_inventory();cfg=pilot_config()['methods'][u['method_id']]
    d=destination(ROOT/'runs'/u['unit_id']);d.mkdir(parents=True,exist_ok=False)
    os.environ.update(isolated_env(d));os.chdir(str(d));records=manifest(u['task_id'],u['unit_id']);write(d/'inputs.json',records)
    costs=dict(load_started=0,load_completed=0,load_cuda_wall_seconds=None,generate_started=0,generate_completed=0,generate_cuda_wall_seconds=0.,
        physical_return_sequences=0,prompt_tokens_unexpanded=0,returned_tokens_through_first_eos=0,inputs_complete=0,evaluated_inputs_complete=0,
        maximum_generate_calls=u['maximum_generate_calls'],property_processes=[],property_events=[],failure=None)
    status=dict(execution_id=APPROVAL,model_id='gellmo_p6_llama',unit=u,status='started',started_unix=time.time(),
        slurm={k:v for k,v in os.environ.items() if k.startswith('SLURM_') or k=='CUDA_VISIBLE_DEVICES'})
    write(d/'started.json',status);t=time.perf_counter()
    def terminate(signum,frame):raise RuntimeError('Slurm termination '+str(signum)+'; no retry')
    signal.signal(signal.SIGTERM,terminate)
    try:
        if u['method_id']=='got':run_got(u,records,d,costs)
        else:
            from recovery_generation import Generation
            gen=Generation(u['method_id'],cfg,costs,d,u['maximum_generate_calls']);gen.start()
            props=LocalProperties(u['task_id'],d,costs,u['method_id'])
            if u['method_id'] in ('beam20','sampling40_prefix20','p6_tot20'):run_flat_or_tot(u,records,gen,props,d,costs)
            elif u['method_id']=='reflexion':run_reflexion(u,records,gen,props,d,costs)
            else:run_terminal(u,records,gen,props,d,costs)
        assert costs['inputs_complete']==costs['evaluated_inputs_complete']==len(records)
        assert costs['generate_started']==costs['generate_completed']<=u['maximum_generate_calls']
        if u['method_id'] not in ('reflexion','s1'):assert costs['generate_completed']==u['maximum_generate_calls']
        assert costs['load_started']==costs['load_completed']==1
        status['status']='complete'
    except BaseException as err:
        status['status']='failed';costs['failure']=repr(err)
        write(d/'exception.json',dict(exception=repr(err),traceback=traceback.format_exc()));raise
    finally:
        status['finished_unix']=time.time();costs['worker_wall_seconds']=time.perf_counter()-t
        if 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized():
            costs['peak_allocated_bytes']=sys.modules['torch'].cuda.max_memory_allocated()
            costs['peak_reserved_bytes']=sys.modules['torch'].cuda.max_memory_reserved()
        write(d/'costs.json',costs);write(d/'status.json',status)


if __name__=='__main__':main()
