"""Review-only property error policy; no model loading, inference or job entrypoint.

An initial RDKit parse success is retained as such. A known downstream
kekulization failure produces an explicit missing property row, never a repaired
molecule, invented score, deleted candidate, or changed original slot.
"""

POLICY_ID = 'rdkit_remove_hs_property_error_isolation_v1'


def property_support_error(mol, chem):
    """Probe the exact failing QED preprocessing operation on a molecule copy.

    The caller handles initial parse-invalid/empty molecules using its existing
    rule. Only KekulizeException is isolated here. Unexpected exceptions still
    abort the worker. No numeric property or molecular fingerprint is computed.
    """
    if mol is None:
        raise ValueError('Initial parse validity must be handled by the caller')
    try:
        chem.RemoveHs(chem.Mol(mol))
    except chem.rdchem.KekulizeException as exc:
        return {
            'policy_id': POLICY_ID,
            'category': 'rdkit_kekulization_after_initial_parse',
            'operation': 'QED.preprocessing.Chem.RemoveHs',
            'exception_type': type(exc).__name__,
            'message': str(exc),
            'initial_parse_valid': True,
            'property_values': None,
            'candidate_repaired': False,
            'numeric_properties_computed_by_probe': False,
        }
    return None
