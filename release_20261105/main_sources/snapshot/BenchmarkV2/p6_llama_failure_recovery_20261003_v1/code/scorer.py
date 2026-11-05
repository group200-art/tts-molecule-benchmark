"""P8-MINMAX-TASK-3OVERM-v1, identical at both normalization stages."""
import math
from collections import Counter
from precision import finite_or_none
from common import M1

VERSION = 'P8-MINMAX-TASK-3OVERM-v1'


def minmax(values, higher_is_better=True):
    values = [finite_or_none(x) for x in values]
    available = [x for x in values if x is not None]
    lo, hi = (min(available), max(available)) if available else (None, None)
    result = []
    for v in values:
        if v is None:
            result.append(0.0)
        elif abs(hi - lo) < 1e-12:
            result.append(0.5)
        else:
            z = (v - lo) / (hi - lo)
            result.append(z if higher_is_better else 1.0 - z)
    return result, {'lo': lo, 'hi': hi, 'finite_mask': [v is not None for v in values],
                    'higher_is_better': higher_is_better, 'raw_values': values}


def score_pool(nodes, property_map, source, objectives):
    assert [n['created_order'] for n in nodes] == list(range(len(nodes)))
    names = [o['property'] for o in objectives]
    assert all(finite_or_none(source.get(p)) is not None for p in names)
    rows = [property_map[n['node_id']] for n in nodes]
    unsupported = []
    for n, r in zip(nodes, rows):
        reason = None
        if r.get('environment_validity_disagreement') and r['rdkit_valid'] and r.get('eval_valid') is False:
            reason = 'unsupported_by_pinned_eval_rdkit'
        if r.get('property_error',{}):
            if r['property_error']['category'] != 'rdkit_kekulization_after_initial_parse':
                raise ValueError('unexpected_property_error:' + n['node_id'])
            reason = 'rdkit_kekulization_after_initial_parse'
        if reason and (r.get('properties') is not None or r.get('raw_properties') is not None):
            raise ValueError('unsupported_must_have_no_properties:' + n['node_id'])
        unsupported.append(reason)
        if not r['rdkit_valid'] and (r.get('properties') is not None or r.get('raw_properties') is not None):
            raise ValueError('invalid_must_have_no_properties:' + n['node_id'])
        if r['rdkit_valid'] and not reason and (not r['inf_canonical'] or
           any(finite_or_none((r.get('properties') or {}).get(p)) is None for p in names + ['sas']) or
           finite_or_none((r.get('raw_properties') or {}).get('sim')) is None):
            raise ValueError('property_unavailable:' + n['node_id'])
    raw = {}
    for o in objectives:
        p = o['property']
        raw[p] = [o['direction'] * (float(r['properties'][p]) - source[p])
                  if finite_or_none((r.get('properties') or {}).get(p)) is not None else None for r in rows]
    raw['similarity'] = [(r.get('raw_properties') or {}).get('sim') for r in rows]
    raw['sas'] = [(r.get('properties') or {}).get('sas') for r in rows]
    ranks, ranges = {}, {}
    for p in names + ['similarity', 'sas']:
        ranks[p], ranges[p] = minmax(raw[p], p != 'sas')
    counts = Counter(r['inf_canonical'] for r in rows if r['inf_canonical'])
    scores = []
    for i, (n, r) in enumerate(zip(nodes, rows)):
        empty = not n['parsed_smiles'].strip()
        valid = r['rdkit_valid']
        penalty = -0.25 if valid and counts[r['inf_canonical']] > 1 else 0.0
        comp = {'validity_component': -20.0 if empty else (1.0 if valid else -10.0)}
        for p in names:
            comp[p + '_impv_rank'] = ranks[p][i]
        comp.update(similarity_rank=ranks['similarity'][i], sas_rank_lower_is_better=ranks['sas'][i], duplicate_penalty=penalty)
        for p in names:
            comp['raw_' + p + '_impv'] = raw[p][i]
        comp.update(raw_similarity=raw['similarity'][i], raw_sas=raw['sas'][i])
        total = 1.0
        for p in names:
            total += (3.0 / len(names)) * ranks[p][i]
        total += 0.50 * ranks['similarity'][i]
        total += 0.25 * ranks['sas'][i]
        total += penalty
        if empty:
            total = -200.0
        elif not valid:
            total = -100.0
        elif unsupported[i]:
            total = -100.0
            comp['property_unavailable_reason'] = unsupported[i]
        assert type(total) is float and math.isfinite(total)
        scores.append({'node_id': n['node_id'], 'value_score': total, 'components': comp})
    order = sorted(range(len(nodes)), key=lambda i: -scores[i]['value_score'])
    for rank, idx in enumerate(order, 1):
        scores[idx]['rank'] = rank
    return {'version': VERSION, 'validity_gate_version': M1, 'scores': scores, 'ranges': ranges,
            'ranked_node_ids': [nodes[i]['node_id'] for i in order]}
