"""Validate nine formal probe batches and report throughput without quality selection."""
import time
from common import *
from replay import Replay
from metrics_entry import evaluate_registered


def main():
    verify()
    assert read(ROOT / 'cpu_checks.json')['status'] == 'PASS'
    assert read(ROOT / 'cpu_property_checks.json')['status'] == 'PASS'
    probes = read(ROOT / 'probe_units.json')
    measured = {m: [] for m in METHODS}
    for index in probes:
        unit = units()[index]
        directory = ROOT / 'runs' / unit['unit_id']
        data = read(directory / 'probe.json.gz')
        assert data['status'] == 'complete' and data['inputs'] == 4 and data['slots'] == 80
        assert data['unit'] == unit and data['plan_sha256'] == sha(ROOT / 'plan.json')
        batch = next(batches(unit))
        generated = read(directory / batch['batch_id'] / 'generation.json.gz')
        assert generated['batch'] == batch and len(generated['lanes']) == 80
        for lane, source in zip(generated['lanes'], batch['lanes']):
            assert lane['record'] == source['record'] and lane['repeat'] == source['repeat']
            assert lane['terminal']['original_slot_id'] == source['repeat']
            if unit['method'] == 's1':
                trace = lane['trajectory']
                assert trace['budget'] == 512 and trace['pre_tokens'] <= 512
                assert trace['wait_eligible'] == trace['wait_applied']
                assert trace['stage_summary']['final']['max_new_tokens'] == 256
            elif unit['method'] == 'reflexion':
                trace = lane['trajectory']
                assert 1 <= len(trace['trials']) <= 3 and trace['trials'][0]['memory_before'] == []
                assert all(not t['feedback']['success'] for t in trace['trials'][:-1])
                assert trace['terminal_trial'] == len(trace['trials']) - 1
                assert trace['stop_reason'] in ('success', 'trial_budget_exhausted')
            else:
                assert len(lane['trajectory']['demo_uids']) == 4
        for row, record in zip(data['rows'], batch['records']):
            assert row == evaluate_registered(unit['task_id'], record['input_uid'], record['source_properties'], row['candidates'], 20)
        measured[unit['method']].append(dict(task=unit['task_id'], wall_seconds=data['wall_seconds'],
             load_seconds=data['model_load_seconds'], gpu_seconds=data['physical_cuda_seconds'],
             feedback_candidates=data['feedback_candidates'], logical_calls=data['logical_calls'], inputs=4))
    first = units()[0]
    first_batch = next(batches(first))
    assert read(ROOT / 'runs' / first['unit_id'] / first_batch['batch_id'] / 'terminals/cache_equivalence.json')['status'] == 'PASS'
    estimates = {}
    for method, samples in measured.items():
        n_units = sum(u['method'] == method for u in units())
        seconds = [(s['wall_seconds'] - s['load_seconds']) / 4 for s in samples]
        # Includes cold property initialization in each measured batch: conservative overhead estimate.
        mean = sum(seconds) / len(seconds)
        largest = max(seconds)
        loads = sum(s['load_seconds'] for s in samples) / len(samples) * n_units
        estimates[method] = dict(observed_seconds_per_input=seconds,
            allocated_gpu_hours_central=(3905 * mean + loads) / 3600,
            allocated_gpu_hours_planning=(3905 * max(mean * 1.5, largest * 1.2) + loads) / 3600)
    result = dict(status='PASS', unix=time.time(), probe_inputs_per_method=12, formal_terminal_slots_reused=720,
                  quality_based_gate=False, measurements=measured, estimates=estimates,
                  two_gpu_wall_hours_central=sum(e['allocated_gpu_hours_central'] for e in estimates.values()) / 2,
                  two_gpu_wall_hours_planning=sum(e['allocated_gpu_hours_planning'] for e in estimates.values()) / 2,
                  limitations='Small timing sample, queues/interruptions excluded; property cold-start repeated in extrapolation; not confidence bounds',
                  plan_sha256=sha(ROOT / 'plan.json'))
    save(ROOT / 'probe_gate.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
