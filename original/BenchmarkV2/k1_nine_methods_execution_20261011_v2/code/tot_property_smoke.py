"""Eight existing pilot endpoints: CPU interface smoke check, never a quality gate.

Outputs stay under runs/tot_property_smoke. They neither replace formal task
exports nor contribute an extra statistical sample. No generation is invoked.
"""
import argparse
import copy
import json
import os
import time
import traceback

from self_refine_common import ROOT, read, sha, write, require_allocation
from self_refine_properties import evaluate_candidates
from evaluate_tot import records, validate_generation, validate_endpoint
from cache_metrics import evaluate_registered


MODELS = ('mistral', 'llama')
SMOKE_TASKS = ('BDP', 'BHMQ')


def pilot_group(model, task_id):
    model_id = 'gellmo_p6_' + model
    contract_sha = sha(ROOT / 'config/tot_contract.json')
    group = []
    for record in records(task_id)[:2]:
        path = ROOT / 'runs' / ('tot_' + model + '_' + task_id) / ('%04d' % record['input_idx']) / 'complete.json'
        generated = read(path)
        selected = validate_generation(generated, record, path, model_id, contract_sha)
        group.append(dict(record=record, generated=generated, selected=selected,
                          source_path=str(path), source_sha256=sha(path)))
    assert len(group) == 2 and [x['record']['input_idx'] for x in group] == [0, 1]
    return group


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate-only', action='store_true',
                        help='Read generation evidence only; no property model or output writes')
    args = parser.parse_args()
    # Verify all eight upstream endpoints before requesting any predictions.
    groups = {(model, task_id): pilot_group(model, task_id)
              for model in MODELS for task_id in SMOKE_TASKS}
    assert sum(len(group) for group in groups.values()) == 8
    if args.validate_only:
        print(json.dumps(dict(status='PASS_EIGHT_COMMITTED_GENERATION_ENDPOINTS',
                              property_evaluation_performed=False, endpoints=8)), flush=True)
        return
    require_allocation()
    directory = ROOT / 'runs/tot_property_smoke'
    started = time.perf_counter()
    status = dict(status='started', job_id=os.environ['SLURM_JOB_ID'], endpoints=8,
                  new_LLM_generation=0, statistical_role='engineering_check_of_existing_pilot_endpoints',
                  score_or_validity_threshold=False, formal_exports_modified=False)
    write(directory / ('started_' + os.environ['SLURM_JOB_ID'] + '.json'), status)
    completed = []
    try:
        for (model, task_id), group in groups.items():
            model_id = 'gellmo_p6_' + model
            output_dir = directory / (model + '_' + task_id)
            source_identity = [dict(path=x['source_path'], sha256=x['source_sha256']) for x in group]
            write(output_dir / 'sources.json', source_identity)
            filled = evaluate_candidates([x['record'] for x in group],
                [copy.deepcopy(x['selected']) for x in group], output_dir / 'properties',
                model_id, property_method='p6_tot20')
            assert len(filled) == len(group) == 2
            endpoints = []
            for candidate, original in zip(filled, group):
                record, generated = original['record'], original['generated']
                assert all(candidate[k] == value for k, value in original['selected'].items()), \
                    'Property evaluation changed an LM-selected generation field'
                assert sha(original['source_path']) == original['source_sha256']
                metric = evaluate_registered(task_id, record['input_uid'], record['source_properties'],
                                             [candidate], 1)
                result = dict(schema_version='BV2-K1-ENDPOINT-v2', model_id=model_id,
                    method_id='tot_lm_vote', task_id=task_id, input_uid=record['input_uid'],
                    input_idx=record['input_idx'], source_properties=record['source_properties'],
                    selected_candidate=candidate, evaluation=metric, costs=generated['costs'],
                    selection_provenance=dict(generated['selection_provenance'],
                        generation_endpoint_sha256=original['source_sha256']))
                validate_endpoint(result, record, model_id)
                # Successful, unsuccessful, empty, and invalid outcomes all pass
                # this engineering gate when the interface is faithfully preserved.
                endpoints.append(result)
            write(output_dir / 'checked_endpoints.json.gz', endpoints)
            completed.append(dict(model_id=model_id, task_id=task_id, endpoints=len(endpoints),
                                  checked_endpoints_sha256=sha(output_dir / 'checked_endpoints.json.gz')))
            print(json.dumps(dict(event='tot_property_smoke_group_complete', model=model,
                                  task_id=task_id, endpoints=len(endpoints))), flush=True)
        assert len(completed) == 4 and sum(x['endpoints'] for x in completed) == 8
        status.update(status='PASS_CPU_PROPERTY_INTERFACE_EIGHT_EXISTING_ENDPOINTS', groups=completed)
        write(directory / 'complete.json', status)
    except BaseException as error:
        status.update(status='failed', error=repr(error), traceback=traceback.format_exc(), groups=completed)
        raise
    finally:
        status['wall_seconds'] = time.perf_counter() - started
        write(directory / ('status_' + os.environ['SLURM_JOB_ID'] + '.json'), status)
        print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
