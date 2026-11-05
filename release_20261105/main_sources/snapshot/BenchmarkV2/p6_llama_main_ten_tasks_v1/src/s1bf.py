"""README-style cumulative token cap with at most one Wait; no model imports."""
from contracts import experiment_id

def checked_result(result,cap):
    ids=result['generated_ids']
    if not ids or len(ids)>cap or any(type(x) is not int for x in ids):
        raise ValueError('Invalid generated IDs or exceeded cap')
    ended=ids[-1]==128009
    if 128009 in ids[:-1]: raise ValueError('Unexpected tokens after EOS in single sequence')
    if result['ended_with_eos']!=ended or result['content_ids']!=(ids[:-1] if ended else ids):
        raise ValueError('EOS/context identity mismatch')
    return result

def run_one_budget(gen,prompt_ids,budget,context):
    exp=experiment_id(budget)
    if not prompt_ids or prompt_ids[0]!=128000: raise ValueError('Original BOS-bearing prompt required')
    ctx=dict(context,experiment_id=exp,budget=budget)
    initial=checked_result(gen.generate_ids(list(prompt_ids),budget,0,dict(ctx,stage='initial')),budget)
    remaining=budget-len(initial['generated_ids'])
    prefix=list(prompt_ids)+initial['content_ids']
    wait_eligible=initial['ended_with_eos'] and remaining>0
    continuation=None
    if wait_eligible:
        prefix=prefix+list(gen.wait_ids)
        continuation=checked_result(gen.generate_ids(prefix,remaining,1,dict(ctx,stage='continuation')),remaining)
        remaining-=len(continuation['generated_ids'])
        prefix=prefix+continuation['content_ids']
    final_prefix=prefix+list(gen.final_cue_ids)
    final=checked_result(gen.generate_ids(final_prefix,256,0,dict(ctx,stage='final')),256)
    stages={'initial':initial,'continuation':continuation,'final':final}
    brief={}
    for name,value in stages.items():
        brief[name]=None if value is None else {k:value[k] for k in
            ('call_index','output_tokens','input_tokens','ended_with_eos','length_stop','cap_exhausted','max_new_tokens','min_new_tokens','cuda_wall_seconds')}
    return {'experiment_id':exp,'budget':budget,'context':context,'original_prompt_ids':list(prompt_ids),
            'wait_ids':list(gen.wait_ids),'final_cue_ids':list(gen.final_cue_ids),
            'wait_eligible':wait_eligible,'wait_applied':wait_eligible,
            'pre_tokens':budget-remaining,'unused_budget':remaining,'pre_cap_reached':remaining==0,
            'final_text':final['text'],'final_generated_ids':final['generated_ids'],
            'stage_summary':brief,'call_indices':[v['call_index'] for v in stages.values() if v is not None],
            'total_output_tokens':budget-remaining+len(final['generated_ids']),
            'total_input_tokens':sum(v['input_tokens'] for v in stages.values() if v is not None),
            'generate_cuda_wall_seconds':sum(v['cuda_wall_seconds'] for v in stages.values() if v is not None)}
