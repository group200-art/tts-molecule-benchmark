"""CPU-only evaluation of already committed LM-selected ToT terminal molecules."""
import argparse
import gzip
import json
import os
from pathlib import Path

from self_refine_common import ROOT, read, sha, write, require_allocation
from self_refine_properties import evaluate_candidates
from cache_metrics import evaluate_registered, aggregate_registered
from tot_protocol import vote_ranking, parse_smiles

TASKS = ['BDP','BDQ','BPQ','DPQ','BDPQ','MPQ','BDMQ','BHMQ','BMPQ','HMPQ']
MODELS = ['mistral', 'llama']


def records(task):
    spec = read(ROOT / 'config/manifests.json')[task]
    assert sha(spec['path']) == spec['sha256']
    rows = [json.loads(line) for line in Path(spec['path']).read_text().splitlines() if line.strip()]
    assert len(rows) == spec['inputs']
    return rows


def validate_generation(g, record, path, model_id, contract_sha):
    """Read selection from generation evidence, never from property success."""
    assert g['status'] == 'complete_generation_pending_property_evaluation'
    assert g['contract_sha256'] == contract_sha
    assert all(g[k] == record[k] for k in ('input_uid','input_idx','task_id','source_smiles','source_properties'))
    assert g['model_id'] == model_id and g['method_id'] == 'tot_lm_vote'
    assert len(g['initial_candidates']) == 5 and len(g['children']) == 15
    assert [c['original_slot_id'] for c in g['initial_candidates']] == list(range(5))
    assert [c['original_slot_id'] for c in g['children']] == list(range(5, 20))
    assert len({c['candidate_id'] for c in g['children']}) == 15
    prov = g['selection_provenance']
    assert prov['source_path'] == str(path) and prov['contract_sha256'] == contract_sha
    assert prov['candidate_property_access'] is False and prov['external_validity_filter'] is False
    calls = {}
    for phase in ('parent_vote', 'children', 'terminal_vote'):
        call_path = path.parent / (phase + '.json')
        call = read(call_path)
        assert call['status'] == 'complete' and call['request']['phase'] == phase
        assert call['request']['contract_sha256'] == contract_sha
        assert 'gellmo_p6_' + call['request']['model'] == model_id
        if phase != 'children':
            assert prov[phase + '_call_path'] == str(call_path)
        calls[phase] = call
    assert g['parent_vote'] == vote_ranking(calls['parent_vote']['outputs'], 5)
    assert g['terminal_vote'] == vote_ranking(calls['terminal_vote']['outputs'], 15)
    assert g['selected_parent_indices'] == g['parent_vote']['ranking'][:3]
    assert len(calls['children']['outputs']) == 15
    for index, (child, output) in enumerate(zip(g['children'], calls['children']['outputs'])):
        parent = g['initial_candidates'][g['selected_parent_indices'][index // 5]]
        assert child['parent_candidate_id'] == parent['candidate_id']
        assert child['parent_original_slot_id'] == parent['original_slot_id']
        assert child['parent_rank'] == index // 5 + 1 and child['branch'] == index % 5
        assert child['raw_text'] == output['raw_text']
        assert child['raw_smiles'] == parse_smiles(output['raw_text'])['parsed_smiles']
        assert all(child[k] == record[k] for k in ('input_uid','input_idx','task_id','source_smiles'))
        assert child['model_id'] == model_id and child['method_id'] == 'tot_lm_vote'
    terminal_index = g['terminal_vote']['ranking'][0]
    chosen = g['children'][terminal_index]
    expected = dict(chosen, source_original_slot_id=chosen['original_slot_id'], original_slot_id=0)
    assert g['selected_candidate'] == expected
    assert prov['selected_terminal_frontier_index'] == terminal_index
    assert prov['selected_candidate_id'] == chosen['candidate_id']
    return expected


def validate_endpoint(row, record, model_id):
    assert row['schema_version'] == 'BV2-K1-ENDPOINT-v2'
    assert row['model_id'] == model_id and row['method_id'] == 'tot_lm_vote'
    assert all(row[k] == record[k] for k in ('input_uid','input_idx','task_id','source_properties'))
    candidate = row['selected_candidate']
    assert candidate['original_slot_id'] == 0 and 5 <= candidate['source_original_slot_id'] < 20
    assert candidate['model_id'] == model_id and candidate['method_id'] == 'tot_lm_vote'
    assert all(candidate[k] == record[k] for k in ('input_uid','input_idx','task_id','source_smiles'))
    expected = evaluate_registered(record['task_id'], record['input_uid'], record['source_properties'], [candidate], 1)
    assert row['evaluation'] == expected, 'Endpoint metric/candidate mismatch'
    assert row['selection_provenance']['selected_candidate_id'] == candidate['candidate_id']
    assert row['selection_provenance']['candidate_property_access'] is False
    assert row['selection_provenance']['external_validity_filter'] is False


def task(model, task_id):
    require_allocation()
    recs = records(task_id)
    contract_sha = sha(ROOT / 'config/tot_contract.json')
    generated, candidates, provenance = [], [], []
    model_id = 'gellmo_p6_' + model
    # Complete generation for the entire task must be present before any evaluation.
    for r in recs:
        path = ROOT / 'runs' / ('tot_' + model + '_' + task_id) / ('%04d' % r['input_idx']) / 'complete.json'
        g = read(path)
        selected = validate_generation(g, r, path, model_id, contract_sha)
        generated.append(g)
        candidates.append(json.loads(json.dumps(selected)))
        provenance.append(dict(g['selection_provenance'], generation_endpoint_sha256=sha(path)))
    directory = ROOT / 'runs' / ('tot_properties_' + model + '_' + task_id)
    filled = evaluate_candidates(recs, candidates, directory, model_id, property_method='p6_tot20')
    assert len(filled) == len(recs)
    endpoints = []
    for r, c, g, prov in zip(recs, filled, generated, provenance):
        assert all(c[k] == value for k, value in g['selected_candidate'].items()), 'Property evaluation altered generation endpoint'
        evaluation = evaluate_registered(task_id, r['input_uid'], r['source_properties'], [c], 1)
        endpoints.append(dict(schema_version='BV2-K1-ENDPOINT-v2', model_id=model_id,
                              method_id='tot_lm_vote', task_id=task_id, input_uid=r['input_uid'],
                              input_idx=r['input_idx'], source_properties=r['source_properties'],
                              selected_candidate=c, selection_provenance=prov, evaluation=evaluation,
                              costs=g['costs']))
        validate_endpoint(endpoints[-1], r, model_id)
    output = ROOT / 'results' / ('tot_' + model + '__' + task_id + '.json.gz')
    write(output, endpoints)
    write(ROOT / 'reports' / ('tot_properties_' + model + '_' + task_id + '.json'),
          dict(status='COMPLETE_TASK_K1', model_id=model_id, task_id=task_id, inputs=len(recs),
               output_path=str(output), output_sha256=sha(output),
               metrics=aggregate_registered([r['evaluation'] for r in endpoints])))
    print(json.dumps(dict(event='tot_task_evaluated', model=model, task=task_id, inputs=len(recs))), flush=True)


def finalize():
    for model in MODELS:
        endpoints, metrics = [], []
        for task_id in TASKS:
            path = ROOT / 'results' / ('tot_' + model + '__' + task_id + '.json.gz')
            rows = read(path)
            recs = records(task_id)
            assert [r['input_uid'] for r in rows] == [r['input_uid'] for r in recs]
            receipt = read(ROOT / 'reports' / ('tot_properties_' + model + '_' + task_id + '.json'))
            assert receipt['status'] == 'COMPLETE_TASK_K1' and receipt['output_sha256'] == sha(path)
            assert receipt['model_id'] == 'gellmo_p6_' + model and receipt['task_id'] == task_id
            assert receipt['inputs'] == len(rows) == len(recs)
            for row, record in zip(rows, recs):
                validate_endpoint(row, record, 'gellmo_p6_' + model)
                prov = row['selection_provenance']
                assert sha(prov['source_path']) == prov['generation_endpoint_sha256']
            endpoints.extend(rows)
            metrics.append(dict(task_id=task_id, model_id='gellmo_p6_' + model, method_id='tot_lm_vote',
                                metrics=aggregate_registered([r['evaluation'] for r in rows])))
        assert len(endpoints) == 3905
        assert len({(r['task_id'], r['input_uid']) for r in endpoints}) == 3905
        blob = ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in endpoints).encode()
        path = ROOT / 'results' / ('tot_' + model + '__tot_lm_vote.jsonl.gz')
        if path.exists():
            with gzip.open(path, 'rb') as f:
                assert f.read() == blob
        else:
            temp = path.with_name('.' + path.name + '.tmp')
            with temp.open('xb') as f:
                with gzip.GzipFile(fileobj=f, mode='wb', mtime=0) as z:
                    z.write(blob)
            temp.replace(path)
        write(ROOT / 'results' / ('tot_' + model + '__group_metrics.json'), metrics)
        write(ROOT / 'reports' / ('tot_' + model + '_complete.json'),
              dict(status='PASS_FULL3905_LM_VOTE_K1', model_id='gellmo_p6_' + model,
                   final_records=3905, output_path=str(path), output_sha256=sha(path)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--finalize', action='store_true')
    parser.add_argument('--unit', type=int)
    args = parser.parse_args()
    if args.finalize:
        finalize()
    else:
        index = args.unit if args.unit is not None else int(os.environ['SLURM_ARRAY_TASK_ID'])
        assert 0 <= index < 20
        task(MODELS[index // 10], TASKS[index % 10])
