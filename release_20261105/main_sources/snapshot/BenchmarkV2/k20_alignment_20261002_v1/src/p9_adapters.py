"""Fixed old P9 molecular prompt and final-only K1 parser."""
import re
from contracts import ROOT, read, task_contract, digest, experiment_id, science
METHOD = 'p9_s1bf_native_k1_v1'

def build_prompt(record):
    p=read(ROOT/'configs/p9_prompt.json')
    objectives=task_contract(record['task_id'])['objectives']
    parts=[('decrease ' if o['direction']==-1 else 'increase ')+p['property_names'][o['property']] for o in objectives]
    if len(parts)==3: adjustment=', '.join(parts[:2])+' and '+parts[2]
    elif len(parts)==4: adjustment=', '.join(parts[:3])+' and '+parts[3]
    else: raise ValueError('Only approved three/four objective tasks')
    return p['instruction']+'\n%%% Input : <SMILES> '+record['source_smiles']+' </SMILES>\n%%% Adjust: '+adjustment+'\n'

def parse_final(text):
    matches=re.findall(science()['parser_regex'],text,flags=re.IGNORECASE|re.DOTALL)
    nonempty=[m.strip() for m in matches if m.strip()]
    return {'raw_smiles':nonempty[-1] if nonempty else '',
            'parser_match_count':len(matches),'parser_nonempty_match_count':len(nonempty)}

def candidate(run_id,record,budget,final_text,trace_path):
    exp=experiment_id(budget)
    identity={'run_id':run_id,'experiment_id':exp,'task_id':record['task_id'],
              'input_uid':record['input_uid'],'original_slot_id':0}
    out=dict(identity,candidate_id='p9-candidate-v1-'+digest(identity),method_id=METHOD,
             input_idx=record['input_idx'],source_smiles=record['source_smiles'],
             source_molecule_uid=record['source_molecule_uid'],final_text=final_text,
             trace_path=trace_path,stage='final',budget=budget)
    out.update(parse_final(final_text))
    return out

def validate_candidates(candidates,records,run_id):
    expected=[(r['input_uid'],b) for r in records for b in (128,512,2048)]
    if [(c['input_uid'],c['budget']) for c in candidates]!=expected:
        raise ValueError('Every UID must retain all three independent final slots in order')
    byuid={r['input_uid']:r for r in records}
    for c in candidates:
        r=byuid[c['input_uid']]
        rebuilt=candidate(run_id,r,c['budget'],c['final_text'],c['trace_path'])
        if c!=rebuilt: raise ValueError('Candidate identity/parser changed')
    if len({c['candidate_id'] for c in candidates})!=len(candidates):
        raise ValueError('Candidate IDs collide')
