"""Unchanged registered native metric functions, copied from NEXT3."""
from contracts import kernel,task_contract
from property_bridge import finite_or_none

def evaluate_registered(task_id, input_uid, source, candidates, expected_slots, objectives=None):
    task = task_contract(task_id, objectives)
    names = [o['property'] for o in task['objectives']]
    if set(source) != set(names):
        raise ValueError('Source property set differs from registered task')
    if [c['original_slot_id'] for c in candidates] != list(range(expected_slots)):
        raise ValueError('Missing or reordered original slot')
    slots = [{'slot_id':c['original_slot_id'], 'smiles':c['raw_smiles'],
              'rdkit_valid':c['rdkit_valid'], 'properties':c['properties']} for c in candidates]
    k = kernel()
    result = k['evaluate_input'](input_uid, source, slots, names,
                                {o['property']:o['supplementary_success_delta'] for o in task['objectives']}, expected_slots)
    # The legacy -1 initialization is harmless for positive goals but can suppress
    # a strictly improving negative Mutag candidate when its source is zero.
    # Preserve the exact score and >= tie order; initialize from the success pool
    # so the registered any-success definition holds on the whole finite domain.
    eligible = [i for i,c in enumerate(candidates) if c['rdkit_valid'] and bool(c['raw_smiles'].strip()) and k['finite_values'](c['properties'],names)]
    directional = [i for i in eligible if k['strictly_improves'](source,candidates[i]['properties'],names)]
    thresholds = {o['property']:o['supplementary_success_delta'] for o in task['objectives']}
    delta = [i for i in directional if k['meets_delta'](source,candidates[i]['properties'],names,thresholds)]
    winners = {}
    for suffix,pool in [('dir',directional),('delta',delta)]:
        best = max(pool,key=lambda i:(k['legacy_selection_score'](source,candidates[i]['properties'],names),i)) if pool else None
        winners[suffix] = best
        result['SR_'+suffix+'_success'] = best is not None
        result['selected_slot_id_'+suffix] = best
    winner = winners['dir']
    result['RI_legacy'] = k['legacy_report_ri'](source,candidates[winner]['properties'],names) if winner is not None else None
    result['RI_abs_eps01'] = k['ri_abs_eps01'](source,candidates[winner]['properties'],names) if winner is not None else None
    result['metric_version'] = 'v2_dir_registered_guarded_legacy_rank_v1'
    result.update({'task_id':task_id, 'interface_version':'BV2-metrics-only-pilot-v1', 'source_properties':source})
    result['candidates'] = candidates
    result['n_property_rows'] = sum(c.get('property_row_exists', c['properties'] is not None) for c in candidates)
    result['n_finite_property_slots'] = sum(k['finite_values'](c['properties'], names) for c in candidates)
    result['selected_property_values'] = candidates[winner]['properties'] if winner is not None else None
    for suffix in ('dir', 'delta'):
        slot = result['selected_slot_id_' + suffix]
        result['selected_candidate_id_' + suffix] = candidates[slot]['candidate_id'] if slot is not None else None
        result['selected_smiles_' + suffix] = candidates[slot]['raw_smiles'] if slot is not None else None
    return result

def aggregate_registered(rows):
    out = kernel()['aggregate'](rows)
    nslots = sum(r['n_slots'] for r in rows)
    nrows = sum(r['n_property_rows'] for r in rows)
    nfinite = sum(r['n_finite_property_slots'] for r in rows)
    out.update({'n_slots':nslots, 'n_property_rows':nrows, 'n_finite_property_slots':nfinite,
                'n_success_delta':sum(r['SR_delta_success'] for r in rows),
                'property_row_coverage_pct':100*nrows/nslots if nslots else None,
                'finite_target_property_coverage_pct':100*nfinite/nslots if nslots else None,
                'legacy_Val_diagnostic':100*sum(r['n_property_rows'] > 0 for r in rows)/len(rows) if rows else None})
    successes=[r for r in rows if r['SR_dir_success']]
    for prop,label in [('sim','Sim_cached'),('sas','SAS_cached')]:
        values=[finite_or_none(r['selected_property_values'].get(prop)) for r in successes]
        out[label+'_available_count']=sum(v is not None for v in values)
        out[label]=sum(values)/len(values) if values and all(v is not None for v in values) else None
    out['auxiliary_basis']='selected-success comparison cache only; null if any selected cache value missing; Phase1D pair Sim/SAS calculated then comparison-round2 cached; no claim of full legacy equivalence'
    names=list(rows[0]['source_properties']) if rows else []
    out['APS']={p:sum(r['selected_property_values'][p] for r in successes)/len(successes) if successes else None for p in names}
    return out
