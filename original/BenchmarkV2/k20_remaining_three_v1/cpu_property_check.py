"""Real CPU backend integration checks while GPUs queue; no generated benchmark outputs."""
import time
from common import *
from property_client import PropertyClient


def main():
    require_allocation()
    verify()
    directory = ROOT / 'cpu_property_check'
    directory.mkdir(exist_ok=False)
    os.environ.update(isolate(directory / 'environment', cpu=True))
    client = PropertyClient(directory)
    started = time.time()
    checks = []
    try:
        for method, task in [('fewshot', 'BDP'), ('s1', 'DPQ'), ('reflexion', 'BHMQ')]:
            record = manifest(task)[0]
            candidates = []
            for repeat in range(20):
                raw = [record['source_smiles'], '', 'not_a_molecule'][repeat % 3]
                lane = dict(record=record, repeat=repeat)
                c = terminal(method, lane, raw, '<SMILES>' + raw + '</SMILES>', dict(diagnostic_only=True))
                c['candidate_id'] = 'diagnostic-' + c['candidate_id']
                candidates.append(c)
            result = client.evaluate(method, task, [record], candidates, 'terminals',
                                     directory / (method + '_' + task), cache_check=(method == 'fewshot'))
            assert len(result['filled']) == 20 and len(result['rows']) == 1
            names = [o['property'] for o in task_contract(task)['objectives']]
            for i, c in enumerate(result['filled']):
                if i % 3 == 0:
                    assert c['rdkit_valid'] and c['property_row_exists']
                    assert all(c['raw_properties'][p] is not None and c['properties'][p] is not None for p in names)
                else:
                    assert not c['rdkit_valid'] and c['properties'] is None
            receipt = read(directory / (method + '_' + task) / 'complete.json')
            inf = read(receipt['inf']['result']['result_path'])
            if method == 's1':
                assert inf['identity']['admet_status'] == 'not_required' and inf['identity']['model_load_calls'] == 0
            checks.append(dict(method=method, task=task, slots=20, valid_empty_invalid_preserved=True,
                               raw_and_round2_complete=True, inf_identity=inf['identity']))
    finally:
        client.close()
    result = dict(status='PASS', checks=checks, wall_seconds=time.time() - started,
                  benchmark_outputs_generated=0, plan_sha256=sha(ROOT / 'plan.json'))
    save(ROOT / 'cpu_property_checks.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
