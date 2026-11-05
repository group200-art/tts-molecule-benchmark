"""Reuse the accepted molecular-error policy in online Reflexion feedback.

No molecule repair, score imputation, slot replacement, prompt change, or extra
trial. Only the already documented RemoveHs/KekulizeException is isolated.
"""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ACCEPTED = REPO / 'BenchmarkV2/p6_llama_failure_recovery_20261003_v1'
POLICY = 'rdkit_remove_hs_property_error_isolation_v1'


def known_error(error):
    if error is None:
        return False
    expected = dict(policy_id=POLICY,
                    category='rdkit_kekulization_after_initial_parse',
                    operation='QED.preprocessing.Chem.RemoveHs',
                    exception_type='KekulizeException', initial_parse_valid=True,
                    property_values=None, candidate_repaired=False,
                    numeric_properties_computed_by_probe=False)
    if (not isinstance(error, dict) or
            any(error.get(k) != v for k, v in expected.items()) or
            not isinstance(error.get('message'), str) or not error['message']):
        raise RuntimeError('Unrecognized property error; cannot assign molecular failure')
    return True


def terminal_candidate(candidate, source_properties):
    """Retain validity, error evidence and all slots in the existing readout."""
    import math
    validity = {r['environment']: r['valid']
                for r in candidate['canonical_representations']}
    error = candidate.get('property_error')
    isolated = known_error(error)
    if isolated:
        assert candidate['raw_properties'] is None and candidate['properties'] is None
        assert candidate['rdkit_valid'] and candidate['raw_smiles'].strip()
    known = isolated or bool(validity['gellmo_inf'] and not validity['gellmo_eval_py36'])
    if (validity['gellmo_inf'] and validity['gellmo_eval_py36'] and
            candidate['raw_smiles'].strip() and not isolated):
        props = candidate['properties'] or {}
        assert all(isinstance(props.get(p), (int, float)) and
                   not isinstance(props[p], bool) and math.isfinite(props[p])
                   for p in source_properties), candidate['candidate_id']
    return dict(candidate, known_molecular_property_failure=known,
                environment_validity_disagreement=validity['gellmo_inf'] != validity['gellmo_eval_py36'],
                property_error=error)


def install(service):
    """Install into an isolated service; original shared source files are intact."""
    if service.MODULES.get('known_molecular_error_policy_installed'):
        return
    for name in ('admet_worker', 'standard_producer'):
        spec = importlib.util.spec_from_file_location('accepted_' + name, str(ACCEPTED / (name + '.py')))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.write = service.save
        module.require_allocation = service.require_allocation
        service.MODULES[name] = module

    import property_bridge
    import reflexion
    original_backfill = property_bridge.backfill
    original_feedback = reflexion.feedback_from_filled

    def backfill(candidates, requests, mapping, responses, contract):
        filled = original_backfill(candidates, requests, mapping, responses, contract)
        byid = {r['request_id']: r for r in responses}
        for candidate in filled:
            response = byid[candidate['request_id']]
            error = response.get('property_error')
            if known_error(error):
                assert response['status'] == 'missing' and response['values'] is None
                assert candidate['rdkit_valid'] and candidate['raw_smiles'].strip()
                assert candidate['raw_properties'] is None and candidate['properties'] is None
                candidate['property_error'] = error
                candidate['known_molecular_property_failure'] = True
        return filled

    def feedback(record, candidate, metric_row):
        if not known_error(candidate.get('property_error')):
            return original_feedback(record, candidate, metric_row)
        task = reflexion.task_contract(record['task_id'])
        names = [objective['property'] for objective in task['objectives']]
        reflexion.kernel()['require_source'](record['source_properties'], names)
        assert candidate['rdkit_valid'] and candidate['raw_smiles'].strip()
        assert candidate['raw_properties'] is None and candidate['properties'] is None
        assert not metric_row['SR_dir_success'] and not metric_row['SR_delta_success']
        # Same binary observation as every other non-successful valid candidate.
        # No error text, property values, molecule repair or diagnostic hint is
        # exposed to the model. The original controller retains its trial cap.
        return dict(success=False, reward=0, output_status='known_molecular_property_failure',
                    observation='Answer is INCORRECT',
                    evaluator='single-candidate v2_dir_registered_guarded_legacy_rank_v1',
                    signal_scope='binary and invalid status only; no property values/deltas/rank/repair hints')

    property_bridge.backfill = backfill
    reflexion.feedback_from_filled = feedback
    service.MODULES['known_molecular_error_policy_installed'] = True
