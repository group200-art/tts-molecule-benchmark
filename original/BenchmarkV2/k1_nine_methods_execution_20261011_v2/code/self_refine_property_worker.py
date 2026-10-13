"""Scope-safe invocation of unchanged accepted CPU property producers."""
import argparse
import importlib.util
from self_refine_common import *


def known_failure(response):
    error = response.get('property_error')
    if error is not None:
        return (error.get('policy_id') == 'rdkit_remove_hs_property_error_isolation_v1'
                and error.get('category') == 'rdkit_kekulization_after_initial_parse'
                and error.get('initial_parse_valid') is True)
    versions = {v['environment']: v for v in response['canonical_representations']}
    return (response.get('environment_validity_disagreement') is True
            and versions['gellmo_inf']['valid'] is True
            and versions['gellmo_eval_py36']['valid'] is False
            and response.get('missing_reason') == 'invalid_in_inf_or_eval')


def normalize(payload):
    from property_bridge import backfill
    from contracts import kernel, task_contract
    filled = backfill(payload['candidates'], payload['requests'], payload['mapping'],
                      payload['responses'], payload['cache_contract'])
    responses = {r['request_id']: r for r in payload['responses']}
    names = [o['property'] for o in task_contract(payload['task_id'])['objectives']]
    finite = kernel()['finite_values']
    for c in filled:
        response = responses[c['request_id']]
        c['property_response_missing_reason'] = response.get('missing_reason')
        c['property_error'] = response.get('property_error')
        c['environment_validity_disagreement'] = response.get('environment_validity_disagreement', False)
        c['known_molecular_property_failure'] = known_failure(response)
        if c['rdkit_valid'] and (not finite(c['raw_properties'], names) or not finite(c['properties'], names)):
            assert c['known_molecular_property_failure'], 'Unknown valid-candidate property loss: ' + c['candidate_id']
            assert c['properties'] is None, 'Do not silently accept partial unknown properties'
    return filled


def main():
    p = argparse.ArgumentParser()
    p.add_argument('role', choices=['admet', 'standard', 'normalize'])
    p.add_argument('--input', required=True)
    p.add_argument('--input-sha', required=True)
    p.add_argument('--admet')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    require_allocation()
    for entry in read(ROOT / 'reports/self_refine_property_dependencies.json'):
        assert sha(entry['path']) == entry['sha256'], 'Frozen property dependency changed: ' + entry['path']
    assert sha(args.input) == args.input_sha
    data = read(args.input)
    if args.role == 'normalize':
        write(args.output, normalize(data))
        return
    source = RECOVERY / ('admet_worker.py' if args.role == 'admet' else 'standard_producer.py')
    sys.path.insert(0, str(RECOVERY / 'src'))
    spec = importlib.util.spec_from_file_location('accepted_producer_' + args.role, str(source))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.write = write
    module.require_allocation = require_allocation
    if args.role == 'admet':
        sys.argv = [str(source), '--requests', args.input, '--output', args.output]
        module.main()
    else:
        output = check_output(Path(args.output) / 'scope').parent
        assert not any(output.iterdir()), 'Property output must start empty'
        admet = read(args.admet)
        assert admet['requests_sha256'] == args.input_sha
        module.produce(data, admet, output)


if __name__ == '__main__':
    main()
