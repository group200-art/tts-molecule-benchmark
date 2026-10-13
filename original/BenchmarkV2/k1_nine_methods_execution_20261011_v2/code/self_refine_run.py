"""Evaluate one full Llama r1 task using only recovered generation prefixes."""
import argparse
import time
import traceback
from self_refine_common import *
from self_refine_properties import evaluate_candidates


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task-index', type=int, default=int(os.environ.get('SLURM_ARRAY_TASK_ID', '0')))
    args = p.parse_args()
    require_allocation()
    assert 0 <= args.task_index < len(TASKS)
    task = TASKS[args.task_index]
    directory = ROOT / 'runs/self_refine_llama' / task
    payload = read(directory / 'prepared.json.gz')
    started = time.perf_counter()
    status = dict(task_id=task, job_id=os.environ['SLURM_JOB_ID'], status='started')
    write(directory / ('started_' + os.environ['SLURM_JOB_ID'] + '.json'), status)
    try:
        filled = evaluate_candidates(payload['records'], payload['candidates'], directory / 'properties',
                                     'gellmo_p6_llama', cached_responses=payload['property_cache_sources'])
        results = []
        for record, candidate, provenance in zip(payload['records'], filled, payload['provenance']):
            assert candidate['original_slot_id'] == 1 and candidate['rep_seed'] == 51001
            results.append(endpoint(record, candidate, 'gellmo_p6_llama', dict(provenance,
                properties_path=str(directory / 'properties/filled_candidates.json.gz'),
                property_phase='offline_terminal_only_no_generation_feedback')))
        write(ROOT / 'results' / ('self_refine_llama__' + task + '.json.gz'), results)
        status.update(status='complete', n_inputs=len(results),
                      n_known_molecular_failures=sum(c['known_molecular_property_failure'] for c in filled))
    except BaseException as error:
        status.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        status['wall_seconds'] = time.perf_counter() - started
        write(directory / ('status_' + os.environ['SLURM_JOB_ID'] + '.json'), status)
        print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
