"""Read and audit saved trajectories; never generate or invoke properties."""
from pathlib import Path
from contracts import ROOT, read, sha, digest, manifest, scientific_identity, generation_directory
from self_refine import call_seed, prompt_for, task_text, candidate


def trajectory_path(task,seed,index):
    if seed==51001 and task in ('BDP','BHMQ') and index<5:
        directory=generation_directory('E01',task=task)
    else:
        matches=[s for s in read(ROOT/'configs/shards.json') if s['task_id']==task and s['rep_seed']==seed and s['start']<=index<s['end']]
        if len(matches)!=1:raise ValueError('Ambiguous trajectory/shard identity')
        directory=generation_directory('E02',index=matches[0]['array_index'])
    return directory/'trajectories'/('input{:04d}.json'.format(index))


def audit_trajectory(path, record, seed, deep=False):
    t=read(path)
    if t['record']!=record or t['rep_seed']!=seed or t['scientific_identity_sha256']!=scientific_identity():
        raise ValueError('Trajectory/source/model/method identity mismatch')
    if not t['complete'] or len(t['answers'])!=5 or len(t['feedbacks'])!=5 or len(t['calls'])!=10:
        raise ValueError('Incomplete fixed ten-call trajectory')
    if t['native_round']!=4 or t['visible_k']!=1 or t['inference_tool_calls']!=0:
        raise ValueError('Changed native output or tool access')
    expected=[('init',0),('feedback',0),('refine',1),('feedback',1),('refine',2),('feedback',2),('refine',3),('feedback',3),('refine',4),('feedback',4)]
    if [(c['role'],c['r']) for c in t['calls']]!=expected:raise ValueError('Incorrect role order or terminal feedback')
    if t['task_text']!=task_text(record):raise ValueError('Original task prompt changed')
    templates=read(ROOT/'configs/prompts.json');answers=[];feedbacks=[]
    for c,(role,r) in zip(t['calls'],expected):
        if c!= (t['feedbacks'][r] if role=='feedback' else t['answers'][r]):raise ValueError('Role/result history mismatch')
        prompt=prompt_for(role,t['task_text'],answers,feedbacks,r,templates)
        if c['seed']!=call_seed(seed,record['task_id'],record['input_uid'],role,r) or c['prompt_digest']!=digest(prompt):
            raise ValueError('Prompt or seed derivation mismatch')
        if deep:
            start=read(c['call_started_path']);end=read(c['call_result_path']);o=end.get('output',{})
            if start['prompt']!=prompt or not end['completed'] or o['decoded_continuation_untrimmed']!=c['raw_text']:
                raise ValueError('Raw/prompt/completion mismatch')
            if start['context']['input_uid']!=record['input_uid'] or start['context']['rep_seed']!=seed or start['seed']!=c['seed']:
                raise ValueError('Call UID/seed mismatch')
            ids=start['input_ids'];seq=o['sequence_ids'];cont=o['continuation_ids']
            cap=400 if role=='feedback' else 250
            if seq!=ids+cont or not 1<=len(cont)<=cap or len(ids)+cap>32768 or start['untruncated'] is not True:
                raise ValueError('Token history/context/sequence mismatch')
            if c['input_tokens']!=len(ids) or c['output_tokens']!=(cont.index(2)+1 if 2 in cont else len(cont)):
                raise ValueError('Token accounting mismatch')
            cfg=start['generation_config_passed']
            wanted=dict(do_sample=True,temperature=.7,top_p=1.,top_k=0,num_beams=1,num_return_sequences=1,max_new_tokens=cap,bos_token_id=1,eos_token_id=2,pad_token_id=2)
            if any(cfg.get(k)!=v for k,v in wanted.items()):raise ValueError('Effective generation config mismatch')
        (feedbacks if role=='feedback' else answers).append(c)
    return t


def collect(task,seed,pilot=False,deep=False):
    trajectories=[];missing=[];bad=[];records=manifest(task)[:5] if pilot else manifest(task)
    for rec in records:
        path=trajectory_path(task,seed,rec['input_idx'])
        if not path.exists():missing.append(rec['input_uid']);continue
        try:
            t=audit_trajectory(path,rec,seed,deep=deep)
            trajectories.append(dict(trajectory=t,path=str(path),sha256=sha(path),size_bytes=path.stat().st_size))
        except Exception as error:bad.append(dict(input_uid=rec['input_uid'],path=str(path),error=repr(error)))
    return trajectories,missing,bad
