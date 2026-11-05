"""Actual pilot orchestration, with explicit backend injection and stage gates."""
import re
import time
from collections import defaultdict
from contracts import ROOT, PHASE1A, sha, digest, pilot_config, task_contract, manifest, legacy_prompter, pure_functions
from property_bridge import requests_for, backfill, finite_or_none

SOURCE_BUILD = 'P6/full500/code/build_p6_tot_full500_source_records.py'
DEPTH1 = 'P6/full500/code/run_p6_tot_full500_depth1_chunk.py'
SELECTOR = 'P6/full500/code/select_p6_tot_lite_depth1_for_depth2_reference.py'
LABELS = {'bbbp':'BBB permeability', 'drd2':'DRD2 inhibition', 'plogp':'penalized logP',
          'qed':'QED', 'hia':'Intestinal adsorption', 'mutagenicity':'Mutagenicity'}


def objectives_text(task_id, depth):
    objectives = task_contract(task_id)['objectives']
    names = [x['property'] for x in objectives]
    if all(x['direction'] == 1 for x in objectives):
        lead = 'Modify the molecule with minimal structural changes to increase:' if depth == 1 else 'Further revise the selected candidate to improve:'
        bullets = ['- '+LABELS[p] for p in names]
    else:
        lead = 'Modify the molecule with minimal structural changes to make the following changes:' if depth == 1 else 'Further revise the selected candidate to make the following changes:'
        bullets = ['- '+('increase ' if x['direction']==1 else 'decrease ')+LABELS[x['property']] for x in objectives]
    return lead + '\n' + '\n'.join(bullets)


def tot_prompt(record, branch, parent=None):
    # Only the goal block changes. The original wrapper and branch suffix are reused.
    if parent is None:
        f = pure_functions(SOURCE_BUILD, ['build_p6_depth1_prompt'])['build_p6_depth1_prompt']
        base = f(record['source_smiles'])
        old = 'Modify the molecule with minimal structural changes to increase:\n- BBB permeability\n- DRD2 inhibition\n- penalized logP\n- QED'
        if base.count(old) != 1:
            raise ValueError('Legacy depth1 goal changed')
        base = base.replace(old, objectives_text(record['task_id'], 1))
        return pure_functions(DEPTH1, ['build_depth1_branch_prompt'])['build_depth1_branch_prompt'](base, branch, 5)
    f = pure_functions(SELECTOR, ['build_depth2_prompt'])['build_depth2_prompt']
    base = f(record['source_smiles'], parent['prompt_parent_smiles'], parent['parent_status'], parent['selection_score'])
    old = 'Further revise the selected candidate to improve:\n- BBB permeability\n- DRD2 inhibition\n- penalized logP\n- QED'
    if base.count(old) != 1:
        raise ValueError('Legacy depth2 goal changed')
    return base.replace(old, objectives_text(record['task_id'], 2))


def flat_prompt(record):
    sample = {'source_smiles':record['source_smiles'], 'properties':record['source_properties'],
              'task':record['task_key'], 'instr_idx':record['instr_idx']}
    return legacy_prompter().generate_prompt(sample, False, 'ins', 'simple', False, 'seen')


def parse_text(text, method):
    if not isinstance(text, str):
        raise ValueError('Raw model text must be a string')
    if method == 'p6_tot20':
        return pure_functions(DEPTH1, ['extract_smiles_text'], {'re':re})['extract_smiles_text'](text)
    response = text.split('%%% Response:')[-1].strip()
    # Exact P0 conversion rule: first nonempty tag, normalize internal whitespace.
    matches = re.findall(r'<SMILES>\s*(.*?)\s*</SMILES>', response, flags=re.IGNORECASE|re.DOTALL)
    valid = [x.strip() for x in matches if x.strip()]
    return {'parsed_smiles':' '.join(valid[0].split()) if valid else '',
            'parsed_smiles_original':valid[0] if valid else '',
            'parse_method':'tag_first_nonempty' if valid else 'no_nonempty_tag', 'n_tag_matches':len(valid)}


def candidate(run_id, method, record, stage, slot, node, raw_text):
    identity = {'run_id':run_id, 'model_id':'gellmo_p6_llama', 'method_id':method,
                'task_id':record['task_id'], 'input_uid':record['input_uid'],
                'stage':stage, 'node':node, 'original_slot_id':slot}
    parsed = parse_text(raw_text, method)
    return dict(identity, candidate_id='candidate-v1-'+digest(identity), raw_text=raw_text,
                raw_smiles=parsed['parsed_smiles'], parse_status=parsed,
                input_idx=record['input_idx'], source_smiles=record['source_smiles'])


def validate_candidates(candidates, records, method, run_id):
    config = pilot_config()['methods'][method]
    budget = config['generated_budget']
    if len(candidates) != len(records)*budget:
        raise ValueError('Missing expected candidate slots; no refill')
    for i, record in enumerate(records):
        group = candidates[i*budget:(i+1)*budget]
        for slot, c in enumerate(group):
            if method != 'p6_tot20':
                stage, node = 'flat', 'slot{}'.format(slot)
            elif slot < 5:
                stage, node = 'depth1', 'b{}'.format(slot)
            else:
                stage, node = 'depth2', 'p{}:b{}'.format((slot-5)//5+1, (slot-5)%5)
            expected = candidate(run_id, method, record, stage, slot, node, c['raw_text'])
            for key, value in expected.items():
                if c.get(key) != value:
                    raise ValueError('Candidate identity/order/parse mismatch: '+key)


def rank_parents(record, filled, chemistry_environment):
    if len(filled) != 5 or [x['original_slot_id'] for x in filled] != list(range(5)):
        raise ValueError('ToT requires exactly five depth1 nodes')
    objectives = task_contract(record['task_id'])['objectives']
    metrics = [(x['property'], x['direction']==1) for x in objectives]+[('sim', True), ('sas', False)]
    rank_norm = pure_functions(SELECTOR, ['rank_norm'])['rank_norm']
    rows = []
    for c in filled:
        representations = [v for v in c['canonical_representations'] if v['environment']==chemistry_environment]
        if len(representations)!=1:
            raise ValueError('Ambiguous parent canonical representation')
        row = dict(c, has_props=c['property_row_exists'], canonical_smiles=representations[0]['smiles'])
        row.update({p:finite_or_none((c['parent_properties'] or {}).get(p)) for p, _ in metrics})
        rows.append(row)
    for prop, higher in metrics:
        for row, norm in zip(rows, rank_norm([r[prop] for r in rows], higher)):
            row[prop+'_ranknorm'] = norm
    for row in rows:
        if not row['rdkit_valid']:
            score = -1000000.0
        elif not row['has_props']:
            score = -1000.0
        else:
            score = 1.0
            for obj in objectives:
                n = row[obj['property']+'_ranknorm']
                if n is not None:
                    score += n
            for prop, weight in [('sim', .25), ('sas', .10)]:
                if row[prop+'_ranknorm'] is not None:
                    score += weight*row[prop+'_ranknorm']
        row['selection_score'] = score
    # Python stable reverse sort preserves original order for a complete tie.
    rows.sort(key=lambda r:(r['selection_score'],r['rdkit_valid'],r['has_props'],r['canonical_smiles']), reverse=True)
    for rank, row in enumerate(rows, 1):
        row['parent_selection_rank'] = rank
        row['selected_for_depth2'] = rank <= 3
        row['fallback_used'] = not row['rdkit_valid']
        row['fallback_reason'] = 'rdkit_invalid_original_parent' if row['fallback_used'] else None
        row['prompt_parent_smiles'] = record['source_smiles'] if row['fallback_used'] else row['raw_smiles']
        row['parent_status'] = ('invalid depth-1 candidate; use original source molecule as fallback parent'
                                if row['fallback_used'] else 'valid depth-1 candidate selected by ToT evaluator')
    return rows


def stage_gate(record, d1, bridge):
    """Requires complete response accounting, including explicit misses, before ranking."""
    filled = backfill(d1, bridge['requests'], bridge['mapping'], bridge['responses'], bridge['cache_contract'])
    answers = {r['request_id']:r for r in bridge['responses']}
    admet_names = {'bbbp','hia','mutagenicity'}
    for c in filled:
        r = answers[c['request_id']]
        vals = dict(c['properties'] or {})
        for name in admet_names:
            if finite_or_none(vals.get(name)) is None:
                vals[name] = finite_or_none(r.get('admet_raw_targets', {}).get(name))
        # Old selector reads tables without any similarity column. Do not inject
        # newly computed final-metric Sim into the unchanged parent scorer.
        vals['sim'] = None
        c['parent_properties'] = vals
    ranked = rank_parents(record, filled, bridge['cache_contract']['chemistry_environment'])
    return {'input_uid':record['input_uid'], 'depth1_digest':digest(d1), 'property_bridge_digest':digest(bridge),
            'ranked':ranked, 'parent_digest':digest(ranked), 'status':'validated_fixture' if bridge['cache_contract']['fixture'] else 'validated_real_property_bridge'}


def depth2_plan(record, d1, bridge, gate):
    expected = stage_gate(record, d1, bridge)
    if gate != expected:
        raise ValueError('Depth2 dependency gate missing/stale/altered')
    plans = []
    for parent in gate['ranked'][:3]:
        rank = parent['parent_selection_rank']
        for branch in range(5):
            plans.append({'stage':'depth2', 'node':'p{}:b{}'.format(rank, branch), 'original_slot_id':5+(rank-1)*5+branch,
                          'seed':601+1000000+record['input_idx']*1000+rank*10+branch,
                          'prompt':tot_prompt(record, branch, parent), 'parent_candidate_id':parent['candidate_id'],
                          'parent_raw_smiles':parent['raw_smiles'], 'prompt_parent_smiles':parent['prompt_parent_smiles'],
                          'fallback_used':parent['fallback_used'], 'fallback_reason':parent['fallback_reason'],
                          'upstream':{'depth1':gate['depth1_digest'], 'properties':gate['property_bridge_digest'], 'parents':gate['parent_digest']}})
    return plans

