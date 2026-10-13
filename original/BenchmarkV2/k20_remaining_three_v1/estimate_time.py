"""Timing analysis only; amortize process/model startup per full 80-input unit."""
import time
from statistics import mean
from common import *


def main():
    verify()
    measured = {m: [] for m in METHODS}
    for index in read(ROOT / 'probe_units.json'):
        unit = units()[index]
        directory = ROOT / 'runs' / unit['unit_id']
        data = read(directory / 'probe.json.gz')
        assert data['status'] == 'complete' and data['inputs'] == 4 and data['slots'] == 80
        attempts = [p.parent for p in directory.glob('worker_*/completed.json')
                    if read(p)['completed_unix'] == data['completed_unix']]
        assert len(attempts) == 1
        attempt = attempts[0]
        start = read(attempt / 'started.json')['unix']
        load_start = read(attempt / 'load_started.json')['unix']
        model_startup = load_start - start + data['model_load_seconds']
        batch = next(batches(unit))
        bdir = directory / batch['batch_id']
        receipts = [read(p) for p in sorted((bdir / 'feedback').glob('eval_*/complete.json'))]
        terminal_receipt = read(bdir / 'terminals/complete.json')
        if 'inf' in terminal_receipt:
            receipts.append(terminal_receipt)
        inf_startup = 0
        for i, receipt in enumerate(receipts):
            inf = receipt['inf']
            identity = read(inf['result']['result_path'])['identity']
            if i == 0:
                # Keep a 0.5s allowance for ordinary request IO/RDKit operations.
                inf_startup += max(0, inf['wall_seconds'] - (identity.get('predict_wall_seconds') or 0) - 0.5)
            elif identity['model_load_calls']:
                inf_startup += identity['load_wall_seconds'] or 0
        cache_audit = sum(read(p)['wall_seconds'] for p in (bdir / 'terminals').glob('attempt_*/cache_check.done.json'))
        startup = model_startup + inf_startup
        warm = data['wall_seconds'] - startup - cache_audit
        assert warm >= data['physical_cuda_seconds'] and startup >= 0
        measured[unit['method']].append(dict(task=unit['task_id'], batch_inputs=4, batch_terminals=80,
            total_seconds=data['wall_seconds'], physical_gpu_seconds=data['physical_cuda_seconds'],
            model_and_import_startup_seconds=model_startup, inf_startup_seconds=inf_startup,
            one_off_cache_check_seconds=cache_audit, warm_batch_seconds=warm,
            conservative_note='First standard-property process startup remains in per-batch work'))
    estimates = {}
    for method, samples in measured.items():
        method_units = [u for u in units() if u['method'] == method]
        n_batches = sum(len(list(batches(u))) for u in method_units)
        remaining = n_batches - len(samples)
        startup = mean(s['model_and_import_startup_seconds'] + s['inf_startup_seconds'] for s in samples)
        warm = mean(s['warm_batch_seconds'] for s in samples)
        slower = max(s['warm_batch_seconds'] for s in samples)
        estimates[method] = dict(full_units=len(method_units), full_batches=n_batches,
            remaining_batches_after_probe=remaining,
            central_allocated_gpu_hours=(remaining * warm + len(method_units) * (startup + 10)) / 3600,
            planning_allocated_gpu_hours=(remaining * max(warm * 1.5, slower * 1.2) + len(method_units) * (startup * 1.2 + 10)) / 3600,
            generation_only_gpu_hours=remaining * mean(s['physical_gpu_seconds'] for s in samples) / 3600)
    result = dict(status='TIMING_ESTIMATE_FROM_NINE_REAL_PROBE_BATCHES', unix=time.time(),
        gpu_type='NVIDIA L40S', max_parallel_gpus=2, probe_inputs_per_method=12,
        anchor='start of full array, with720 probe output slots already reusable',
        central_two_gpu_wall_hours=sum(e['central_allocated_gpu_hours'] for e in estimates.values()) / 2,
        planning_two_gpu_wall_hours=sum(e['planning_allocated_gpu_hours'] for e in estimates.values()) / 2,
        measurements=measured, estimates=estimates,
        startup_accounting='Model/import and first inf-process startup charged once per80-input unit; regular scoring charged per4-input batch; extra10s launcher allowance per unit',
        limitations='Small fixed timing sample, not confidence bounds; later task difficulty/early-stopping proportions and shared-cluster queues can change completion time',
        additional_queue_time_included=False, exact_finish_date_guaranteed=False,
        analysis_source_sha256=sha(Path(__file__).resolve()), scientific_plan_sha256=sha(ROOT / 'plan.json'))
    save(ROOT / 'time_estimate.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
