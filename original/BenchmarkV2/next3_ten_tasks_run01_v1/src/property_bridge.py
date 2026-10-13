"""Explicit cached request/response bridge. No predictor imports or fallbacks."""
import csv
import io
import math
import sys
from contracts import digest, unique_pairs

ALIASES = {'BBB_Martins': 'bbbp', 'HIA_Hou': 'hia', 'AMES': 'mutagenicity'}
NAMES = {'bbbp', 'hia', 'mutagenicity', 'drd2', 'qed', 'plogp', 'sas', 'sim'}


def map_fields(fields):
    if fields is None:
        return None
    pairs = []
    for name, value in fields.items():
        canonical = ALIASES.get(name, name)
        if canonical not in NAMES:
            raise ValueError('Unknown property/alias (percentiles and ADMET logP forbidden): ' + name)
        pairs.append((canonical, value))
    return unique_pairs(pairs)


def finite_or_none(value):
    try:
        x = float(value)
        return x if not isinstance(value, bool) and math.isfinite(x) else None
    except (TypeError, ValueError, OverflowError):
        return None


def rounded_csv(fields):
    """Exactly pandas float DataFrame.round(2) -> to_csv -> read_csv.

    Raw values are returned separately by the caller. Only this normalization
    operation imports pandas/NumPy; it computes no molecular properties.
    """
    import numpy as np
    import pandas as pd
    if sys.version_info[:2] != (3, 6) or pd.__version__ != '1.0.5' or np.__version__ != '1.18.5':
        raise RuntimeError('Compatibility normalization requires gellmo_eval_py36 pandas1.0.5/numpy1.18.5')
    names = list(fields)
    if not names:
        return {}, '', {'pandas':pd.__version__, 'numpy':np.__version__}
    values = {p: finite_or_none(v) for p, v in fields.items()}
    frame = pd.DataFrame([values], columns=names, dtype=float).round(2)
    stream = io.StringIO()
    frame.to_csv(stream, index=False)
    serialized = stream.getvalue()
    restored = pd.read_csv(io.StringIO(serialized))
    return {p: finite_or_none(restored.iloc[0][p]) for p in names}, serialized, {'pandas':pd.__version__, 'numpy':np.__version__}


def normalize_response(response):
    r = dict(response)
    precision = r['cache_contract']['precision']
    fields = map_fields(r.get('values'))
    if precision == 'raw_then_pandas_numpy_round2_csv_v1':
        r['raw_properties'] = fields
        if fields is None:
            comparison, csv_text, versions = None, None, None
        else:
            comparison, csv_text, versions = rounded_csv(fields)
        r['comparison_properties'] = comparison
        r['rounding_csv'] = csv_text
        r['rounding_versions'] = versions
        r['raw_availability'] = 'available' if fields is not None else 'missing_property_row'
    elif precision == 'legacy_round2_csv_only_v1':
        r['raw_properties'] = None
        r['raw_availability'] = 'unavailable_already_rounded_cache'
        r['comparison_properties'] = fields
        r['rounding_csv'] = None
        r['rounding_versions'] = None
    else:
        raise ValueError('Unknown precision contract')
    return r


def validate_cache_contract(contract):
    if set(contract) != {'calculators', 'preprocessing', 'precision', 'chemistry_environment', 'fixture'}:
        raise ValueError('Incomplete cache identity')
    if not contract['preprocessing'] or not contract['chemistry_environment'] or type(contract['fixture']) is not bool:
        raise ValueError('Missing cache preprocessing/environment/fixture identity')
    if not isinstance(contract['calculators'], dict) or not contract['calculators']:
        raise ValueError('Explicit calculator/model versions required')
    for key, value in contract['calculators'].items():
        if key not in NAMES or not isinstance(value, str) or not value:
            raise ValueError('Unknown calculator or missing version')
    if contract['precision'] not in ('raw_then_pandas_numpy_round2_csv_v1', 'legacy_round2_csv_only_v1'):
        raise ValueError('Unknown precision contract')


def requests_for(candidates, contract):
    validate_cache_contract(contract)
    requests, mapping, candidate_ids = {}, [], set()
    for c in candidates:
        if c['candidate_id'] in candidate_ids:
            raise ValueError('Duplicate candidate identity')
        candidate_ids.add(c['candidate_id'])
        # Similarity is pair-dependent: include original source identity.
        key = {'raw_smiles':c['raw_smiles'], 'source_smiles':c['source_smiles'], 'cache_contract':contract}
        rid = 'prop-request-v1-' + digest(key)
        if rid in requests and requests[rid]['identity'] != key:
            raise ValueError('Request identity collision')
        requests[rid] = {'request_id':rid, 'identity':key}
        mapping.append({'candidate_id':c['candidate_id'], 'request_id':rid})
    return list(requests.values()), mapping


def backfill(candidates, requests, mapping, responses, contract):
    expected_requests, expected_mapping = requests_for(candidates, contract)
    rmap = unique_pairs([(r['request_id'], r) for r in requests])
    emap = {r['request_id']:r for r in expected_requests}
    if rmap != emap or mapping != expected_mapping:
        raise ValueError('Ambiguous or mismatched request-to-slot mapping')
    answers = unique_pairs([(r['request_id'], r) for r in responses])
    if set(answers) != set(rmap):
        raise ValueError('Missing/extra response: explicitly record cache miss; never infer properties')
    normalized = {}
    for rid, r in answers.items():
        identity = rmap[rid]['identity']
        if r['raw_smiles'] != identity['raw_smiles'] or r['source_smiles'] != identity['source_smiles'] or r['cache_contract'] != contract:
            raise ValueError('Conflicting property response identity')
        if type(r.get('rdkit_valid')) is not bool:
            raise ValueError('Explicit chemistry validity required')
        if not r.get('chemistry_basis'):
            raise ValueError('Missing chemistry basis')
        if r.get('status') not in ('hit', 'missing') or (r['status'] == 'missing' and r.get('values') is not None):
            raise ValueError('Cache status conflict')
        canon = unique_pairs([(x['environment'], x) for x in r['canonical_representations']])
        if contract['chemistry_environment'] not in canon:
            raise ValueError('Missing selector chemistry environment')
        if any(not x.get('rdkit_version') or not isinstance(x.get('smiles'), str) for x in canon.values()):
            raise ValueError('Missing canonical version/string')
        values = map_fields(r.get('values'))
        if values is not None and not set(values).issubset(contract['calculators']):
            raise ValueError('Values lack calculator/model version identity')
        normalized[rid] = normalize_response(r)
    out = []
    for c, m in zip(candidates, mapping):
        r = normalized[m['request_id']]
        item = dict(c)
        item.update({'request_id':m['request_id'], 'rdkit_valid':r['rdkit_valid'],
                     'canonical_representations':r['canonical_representations'], 'chemistry_basis':r['chemistry_basis'],
                     'property_row_exists':r.get('values') is not None, 'cache_status':r['status'],
                     'raw_properties':r['raw_properties'], 'raw_availability':r['raw_availability'],
                     'properties':r['comparison_properties'], 'rounding_csv':r['rounding_csv'],
                     'rounding_versions':r['rounding_versions']})
        out.append(item)
    return out
