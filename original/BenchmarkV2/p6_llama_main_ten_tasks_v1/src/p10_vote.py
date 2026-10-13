"""Exact molecular identity and causal prefix votes; no property/scorer access."""
import hashlib
import time
from collections import Counter
from contracts import BUDGETS,digest
from p10_parser import normalize_raw_text


def classify_ev(parsed_candidate,raw_response):
    from rdkit import Chem,rdBase
    if rdBase.rdkitVersion!='2024.03.6': raise RuntimeError('EV requires RDKit 2024.03.6')
    start=time.perf_counter();counts={'MolFromSmiles':0,'MolToSmiles':0}
    canonical='';reason=None;error=None;candidate=parsed_candidate.strip()
    if not candidate: reason='empty_parsed_candidate'
    else:
        counts['MolFromSmiles']+=1
        try:mol=Chem.MolFromSmiles(candidate,sanitize=True)
        except MemoryError:raise
        except Exception as exc:mol=None;error=repr(exc)
        if mol is None:reason='rdkit_parse_failed'
        elif mol.GetNumAtoms()==0:reason='zero_atom_molecule'
        else:
            counts['MolToSmiles']+=1
            try:canonical=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True,kekuleSmiles=False)
            except MemoryError:raise
            except Exception as exc:reason='canonical_smiles_failed';error=repr(exc)
            if reason is None and not canonical:reason='empty_canonical_smiles'
            if reason is None:
                counts['MolFromSmiles']+=1
                try:again=Chem.MolFromSmiles(canonical,sanitize=True)
                except MemoryError:raise
                except Exception as exc:again=None;error=repr(exc)
                if again is None or again.GetNumAtoms()==0:reason='canonical_roundtrip_failed'
    valid=reason is None
    if not valid:canonical=''
    kind='VALID_CANONICAL' if valid else 'INVALID_RAW'
    value=canonical if valid else normalize_raw_text(raw_response)
    return {'rdkit_valid_ev':valid,'canonical_smiles':canonical,'failure_reason':reason,'identity_exception':error,
            'class_type':kind,'class_value':value,'class_value_sha256':hashlib.sha256(value.encode()).hexdigest(),
            'identity_tool_calls':counts,'identity_wall_seconds':time.perf_counter()-start,'rdkit_version':rdBase.rdkitVersion}


def prefix_vote(rows,budgets=BUDGETS):
    """Consume only as far as the last requested prefix. One physical slot, one vote."""
    if not budgets or any(n not in BUDGETS for n in budgets) or list(budgets)!=sorted(set(budgets)):raise ValueError('Unapproved prefix')
    counts=Counter();first={};valid=0;empty=0;winners=[]
    iterator=iter(rows)
    for slot in range(max(budgets)):
        try:row=next(iterator)
        except StopIteration:break
        if row['global_slot']!=slot or row['original_slot_id']!=slot:raise ValueError('Missing/reordered original vote slot')
        key=(row['class_type'],row['class_value']);counts[key]+=1
        first.setdefault(key,row);valid+=int(row['rdkit_valid_ev']);empty+=int(not row['raw_response'])
        n=slot+1
        if n not in budgets:continue
        keys=sorted(counts,key=lambda k:(-counts[k],first[k]['global_slot'],k))
        win=keys[0];rep=first[win];top=counts[win]
        classes=[{'class_type':k[0],'class_value':k[1],'count':counts[k],'first_global_slot':first[k]['global_slot'],
                  'representative_candidate_id':first[k]['candidate_id']} for k in keys]
        winners.append({'N_used':n,'visible_k':1,'representative':dict(rep),'class_count':len(keys),'top_votes':top,
                        'tie_count':sum(v==top for v in counts.values()),'runner_up_votes':counts[keys[1]] if len(keys)>1 else 0,
                        'valid_slots':valid,'invalid_slots':n-valid,'empty_raw_slots':empty,'classes':classes,
                        'prefix_class_digest':digest(classes),'physical_upstream_calls':max(1,n//20),
                        'physical_upstream_proposals':20*max(1,n//20)})
    return winners


def native_projection(winner,record):
    c=dict(winner['representative'])
    if c['input_uid']!=record['input_uid'] or c['global_slot']>=winner['N_used']:raise ValueError('Representative outside input/prefix')
    c.update(original_slot_id=0,proposal_original_slot_id=c['original_slot_id'],proposal_global_slot=c['global_slot'],
             raw_smiles=c['parsed_candidate'],source_smiles=record['source_smiles'],N_used=winner['N_used'],
             native_view_only=True)
    return c


def prefix_cost(calls,budget):
    if budget not in BUDGETS:raise ValueError('Unapproved cost prefix')
    needed=max(1,budget//20);c=list(calls[:needed])
    if len(c)!=needed or [r['block'] for r in c]!=list(range(needed)) or not all(r['completed'] for r in c):return None
    return {'N_used':budget,'physical_upstream_proposals':needed*20,'calls':needed,
            'generate_cuda_wall_seconds':sum(r['cuda_wall_seconds'] for r in c),
            'tokens_through_first_eos':sum(r['tokens_through_first_eos'] for r in c),
            'prompt_tokens_unexpanded':sum(r['prompt_tokens_unexpanded'] for r in c),
            'prompt_tokens_expanded':20*sum(r['prompt_tokens_unexpanded'] for r in c),
            'complete_shared_upstream_per_input':{'calls':20,'proposals':400},
            'job_overhead_prorated':False,'basis':'Actual complete prefix calls of shared N400 trajectory; N1 owns entire first call'}
