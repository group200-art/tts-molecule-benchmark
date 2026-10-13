"""Self-Refine NeurIPS 2023 Algorithm 1 / Eq.4, C1 molecular adaptation.

Independent paper-algorithm implementation. No property/model imports or tools.
Feedback is always executed before the fixed-timestep stop, including fb4.
"""
import hashlib
import re
from contracts import ROOT, digest, read, legacy_prompter, scientific_identity
METHOD = 'p5_self_refine_neurips23_alg1_p6_v1'


def call_seed(rep_seed, task_id, input_uid, role, r):
    if rep_seed not in (51001,51002,51003) or role not in ('init','feedback','refine'):
        raise ValueError('Unapproved seed/role')
    if type(r) is not int or not 0 <= r <= 4 or (role == 'init' and r != 0) or (role == 'refine' and r == 0):
        raise ValueError('Wrong role round')
    literal = 'P5-SR-A1-v1|{}|{}|{}|{}|{}'.format(rep_seed,task_id,input_uid,role,r)
    return int.from_bytes(hashlib.sha256(literal.encode('utf-8')).digest()[:8],'big') % 2147483647


def task_text(record):
    # Original source properties are deliberately not passed into the prompter.
    sample = dict(source_smiles=record['source_smiles'],properties={},
                  task=record['task_key'],instr_idx=record['instr_idx'])
    full = legacy_prompter().generate_prompt(sample,False,'ins','simple',False,'seen')
    return full.rstrip('\n')


def prompt_for(role, task, answers, feedbacks, r, templates):
    if role == 'init':
        if answers or feedbacks or r != 0: raise ValueError('Bad initial history')
        return templates['init'].format(task=task)
    if role == 'feedback':
        if len(answers) != r+1 or len(feedbacks) != r: raise ValueError('Bad feedback history')
        return templates['feedback'].format(task=task,answer=answers[-1]['raw_text'],r=r)
    if role == 'refine':
        if len(answers) != r or len(feedbacks) != r: raise ValueError('Bad refinement history')
        history = '\n\n'.join('Answer {}:\n{}\n\nFeedback {}:\n{}'.format(
            i,answers[i]['raw_text'],i,feedbacks[i]['raw_text']) for i in range(r))
        return templates['refine'].format(task=task,history=history,r=r)
    raise ValueError('Unknown role')


def parse_answer(text):
    matches = list(re.finditer(r'<SMILES>(.*?)</SMILES>', text, flags=re.S))
    value = matches[0].group(1).strip() if matches else ''
    return dict(raw_smiles=value,n_complete_tags=len(matches),
                status='no_tag' if not matches else 'empty_tag' if not value else 'tag_parsed',
                rule='first_complete_case_sensitive_SMILES_tag; strip_edges_only; no_repair')


def candidate(record, seed, r, answer, identity=None):
    identity = scientific_identity() if identity is None else identity
    ident = dict(method_id=METHOD,model_id='gellmo_p6_llama',task_id=record['task_id'],
                 input_uid=record['input_uid'],rep_seed=seed,original_slot_id=r,
                 scientific_identity_sha256=identity,raw_answer_sha256=digest(answer['raw_text']),
                 generation_seed=call_seed(seed,record['task_id'],record['input_uid'],'init' if r==0 else 'refine',r))
    parsed = parse_answer(answer['raw_text'])
    return dict(ident,candidate_id='p5-sr-candidate-v1-'+digest(ident),input_idx=record['input_idx'],
                source_smiles=record['source_smiles'],raw_text=answer['raw_text'],
                raw_smiles=parsed['raw_smiles'],parse=parsed,call_index=answer['call_index'])


def run_input(record, seed, model_call, emit, templates=None, task=None, identity=None):
    templates = read(ROOT/'configs/p5_prompts.json') if templates is None else templates
    task = task_text(record) if task is None else task
    identity = scientific_identity() if identity is None else identity
    answers, feedbacks, calls = [], [], []
    def invoke(role, r):
        prompt = prompt_for(role,task,answers,feedbacks,r,templates)
        context = dict(task_id=record['task_id'],input_uid=record['input_uid'],
                       input_idx=record['input_idx'],rep_seed=seed,role=role,r=r,
                       input_call_index=len(calls),scientific_identity_sha256=identity)
        derived = call_seed(seed,record['task_id'],record['input_uid'],role,r)
        value = model_call(role,prompt,context,derived)
        if not isinstance(value['raw_text'],str): raise TypeError('Raw response must be text')
        value = dict(value,role=role,r=r,seed=derived,prompt_digest=digest(prompt))
        emit(role,r,value)
        calls.append(value)
        return value
    answers.append(invoke('init',0))
    for r in range(5):
        feedbacks.append(invoke('feedback',r))
        if r == 4: break
        answers.append(invoke('refine',r+1))
    return dict(method_id=METHOD,model_id='gellmo_p6_llama',record=record,
                rep_seed=seed,scientific_identity_sha256=identity,
                task_text=task,answers=answers,feedbacks=feedbacks,calls=calls,
                complete=True,physical_calls=10,generated_molecular_slots=5,
                native_round=4,visible_k=1,inference_tool_calls=0)


def prefix_cost(trajectory, r):
    if type(r) is not int or not 0 <= r <= 4: raise ValueError('Unknown endpoint')
    calls = trajectory['calls'][:2*(r+1)]
    if len(calls) != 2*(r+1) or calls[-1]['role'] != 'feedback' or calls[-1]['r'] != r:
        raise ValueError('Endpoint must include terminal feedback')
    return dict(r=r,model_calls=len(calls),generated_slots=r+1,visible_k=1,
                input_tokens=sum(c['input_tokens'] for c in calls),
                output_tokens=sum(c['output_tokens'] for c in calls),
                generate_cuda_wall_seconds=sum(c['cuda_wall_seconds'] for c in calls),
                independent_slurm_gpu_seconds=None,independent_queue_seconds=None)
