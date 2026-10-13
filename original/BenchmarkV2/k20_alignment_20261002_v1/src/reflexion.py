"""Reflexion CoT molecular port. No model/property imports.

Control flow and prompt structure derived from Noah Shinn's MIT-licensed
hotpotqa_runs/agents.py CoTAgent, REFLEXION branch and CotQA_no_context driver.
See sources/author/snapshot/LICENSE and reports/METHOD_SOURCE_AND_NOVELTY.md.
"""
import re
from contracts import digest, legacy_prompter, task_contract, kernel

METHOD = 'ppro_reflexion_cot_molecular_v1'


def task_text(record):
    """Retain existing seen property wording and Input/Adjust molecular fields.

    The old output-only instruction is replaced by the author's two-stage CoT
    instruction, so it does not contradict the Thought stage.
    """
    sample=dict(source_smiles=record['source_smiles'],properties=record['source_properties'],
                task=record['task_key'],instr_idx=record['instr_idx'])
    prompter=legacy_prompter()
    full=prompter.generate_prompt(sample,False,'ins','simple',False,'seen')
    instruction,separator,body=full.partition('\n')
    if not separator or instruction!=prompter.instructions[record['instr_idx']]:
        raise ValueError('Existing seen prompt boundary changed')
    return body.rstrip()


def parse_action(text):
    matches=list(re.finditer(r'<SMILES>\s*(.*?)\s*</SMILES>',text,flags=re.I|re.S))
    raw=matches[0].group(1).strip() if matches else ''
    return dict(parsed_smiles=raw,parse_status=('no_tag' if not matches else 'empty_tag' if not raw else 'tag_parsed'),
                n_tag_matches=len(matches),parse_rule='first_complete_tag_in_Action_only; strip_edges; no punctuation/case/bracket changes or chemical repair')


def candidate(run_id,record,trial,raw_text,call_index):
    ident=dict(run_id=run_id,model_id='gellmo_p6_mistral',method_id=METHOD,task_id=record['task_id'],
               input_uid=record['input_uid'],stage='trial'+str(trial),original_slot_id=trial)
    parsed=parse_action(raw_text)
    return dict(ident,candidate_id='candidate-v1-'+digest(ident),input_idx=record['input_idx'],
                source_smiles=record['source_smiles'],raw_text=raw_text,raw_smiles=parsed['parsed_smiles'],
                parse=parsed,call_index=call_index,trial_index=trial)


def feedback_from_filled(record,filled,metric_row):
    """Binary external feedback only; predictor/data failures must propagate."""
    task=task_contract(record['task_id'])
    names=[x['property'] for x in task['objectives']]
    k=kernel();k['require_source'](record['source_properties'],names)
    reps=filled['canonical_representations']
    valid_in_eval=next(r['valid'] for r in reps if r['environment']=='gellmo_eval_py36')
    nonempty=bool(filled['raw_smiles'].strip())
    valid=filled['rdkit_valid'] and valid_in_eval and nonempty
    if valid and (not k['finite_values'](filled['raw_properties'],names) or not k['finite_values'](filled['properties'],names)):
        raise RuntimeError('Finite complete target properties missing for a chemically valid output; not reward=0')
    success=valid and metric_row['SR_dir_success']
    if not nonempty:state='empty_output'
    elif not filled['rdkit_valid']:state='invalid_in_generation_property_environment'
    elif not valid_in_eval:state='invalid_in_evaluation_environment'
    else:state='valid_output'
    observation='Answer is CORRECT' if success else 'Answer is INCORRECT'
    if state!='valid_output':observation+='; molecular output is '+('empty' if not nonempty else 'invalid')
    return dict(success=bool(success),reward=int(success),output_status=state,observation=observation,
                evaluator='single-candidate v2_dir_registered_guarded_legacy_rank_v1',
                signal_scope='binary and invalid status only; no property values/deltas/rank/repair hints')


def run_input(record,run_id,templates,model_call,evaluate,emit,question=None):
    """One fresh episodic memory; callbacks expose every physical role/phase.

    model_call(role, prompt, context) returns raw_text and call_index.
    evaluate(candidate) returns full evidence plus its restricted feedback dict.
    """
    memory=[];trials=[];calls=0
    question=task_text(record) if question is None else question
    original_source=record['source_smiles'];original_uid=record['input_uid']

    def invoke(role,prompt,trial):
        nonlocal calls
        if calls>=8:raise RuntimeError('Per-input physical call cap exceeded')
        calls+=1
        return model_call(role,prompt,dict(input_uid=original_uid,input_idx=record['input_idx'],
                          task_id=record['task_id'],source_smiles=original_source,trial_index=trial,
                          role=role,input_call_index=calls-1,memory_digest=digest(memory)))

    for trial in range(3):
        before=list(memory)
        reflections=(templates['experience_header']+'Reflections:\n- '+'\n- '.join(r.strip() for r in memory)) if memory else ''
        # Each new trial resets this short-term trajectory, retaining only reflections.
        scratchpad='\nThought:'
        emit(trial,'started',dict(input_uid=original_uid,source_smiles=original_source,
                                 memory_before=before,scratchpad_before=scratchpad,question=question))
        prompt=templates['actor'].format(reflections=reflections,question=question,scratchpad=scratchpad)
        thought=invoke('thought',prompt,trial);emit(trial,'thought',thought)
        scratchpad+=' '+thought['raw_text'].strip()+'\nAction:'
        prompt=templates['actor'].format(reflections=reflections,question=question,scratchpad=scratchpad)
        action=invoke('action',prompt,trial)
        c=candidate(run_id,record,trial,action['raw_text'],action['call_index'])
        emit(trial,'action',dict(action=action,candidate=c))
        scratchpad+=' '+action['raw_text'].strip()
        outcome=evaluate(c)  # Infrastructure and prediction errors are never converted to reward=0.
        feedback=outcome['feedback']
        if type(feedback.get('success')) is not bool or feedback.get('reward')!=int(feedback['success']):
            raise ValueError('Invalid binary evaluator feedback')
        scratchpad+='\nObservation: '+feedback['observation']
        emit(trial,'evaluation',outcome)
        stop=feedback['success'] or trial==2
        reflection=None
        if not stop:
            prompt=templates['reflection'].format(question=question,scratchpad=scratchpad)
            reflection=invoke('reflection',prompt,trial)
            memory.append(reflection['raw_text'].strip())
            memory=memory[-3:]
            emit(trial,'reflection',dict(reflection=reflection,memory_after=list(memory)))
        item=dict(trial_index=trial,candidate=c,thought=thought,action=action,feedback=feedback,
                  evaluation=outcome,memory_before=before,memory_after=list(memory),
                  reflection=reflection,short_term_trajectory=scratchpad,
                  stop_reason='success' if feedback['success'] else 'trial_budget_exhausted' if stop else None)
        trials.append(item);emit(trial,'completed',item)
        if stop:
            return dict(input_uid=original_uid,input_idx=record['input_idx'],source_smiles=original_source,
                        native_candidate_id=c['candidate_id'],native_candidate=c,terminal_trial=trial,
                        stop_reason=item['stop_reason'],trials=trials,memory=list(memory),physical_calls=calls)
    raise AssertionError('Finite trial loop must return')
