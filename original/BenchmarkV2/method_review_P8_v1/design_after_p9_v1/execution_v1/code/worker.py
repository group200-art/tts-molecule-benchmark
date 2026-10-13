"""One authorized logical shard, all three independent G budgets, exactly once."""
import os
import signal
import sys
import time
import traceback
from common import ROOT, EXECUTION_ID, configure_environment, require_allocation, verify_execution, unit, unit_key, unit_inputs, read, write, utc


def main(stage, index):
    require_allocation()
    verify_execution()
    u = unit(stage, index)
    assert read(ROOT / 'control' / ('submission_intent_' + stage + '.json'))['stage'] == stage
    receipt = ROOT / 'control' / ('submitted_' + stage + '.json')
    if receipt.exists():
        assert read(receipt)['job_id'] == os.environ.get('SLURM_ARRAY_JOB_ID')
    directory = ROOT / 'workers' / unit_key(u)
    directory.mkdir(exist_ok=False)
    configure_environment(directory / 'cache')
    records = unit_inputs(u)
    status = {'execution_id': EXECUTION_ID, 'unit': u, 'job_id': os.environ.get('SLURM_JOB_ID'),
              'array_job_id': os.environ.get('SLURM_ARRAY_JOB_ID'), 'array_task_id': os.environ.get('SLURM_ARRAY_TASK_ID'),
              'host': os.uname().nodename, 'start_utc': utc(), 'status': 'running', 'completed_graphs': [],
              'exception': None, 'wall_seconds': None}
    assert os.environ.get('SLURM_ARRAY_TASK_ID') == str(index)
    assert int(os.environ.get('SLURM_RESTART_COUNT', '0')) == 0
    assert int(os.environ.get('SLURM_CPUS_PER_TASK', '0')) == 8
    assert os.environ.get('SLURM_GPUS_ON_NODE') == '1'
    assert os.environ.get('SLURM_JOB_ACCOUNT') == 'rc_hzh_pi' and os.environ.get('SLURM_JOB_PARTITION') == 'hzhlab'
    write(directory / 'unit_status.json', status)
    write(directory / 'inputs.json', records)
    start = time.perf_counter()
    backend = bridge = None
    active = None

    def terminate(signum, frame):
        raise RuntimeError('Slurm signal {}; preserve partial, never retry'.format(signum))
    signal.signal(signal.SIGTERM, terminate)

    try:
        from generation import Backend
        from property_service import PropertyBridge
        from graph_adapter import MolecularGraph
        backend = Backend(directory, u)
        initialization_started = time.perf_counter()
        try:
            backend.start()
        finally:
            backend.cost['generation_initialization_seconds'] = time.perf_counter() - initialization_started
        bridge = PropertyBridge(u['task_id'], directory)
        bridge.start()
        for G in [12, 6, 24]:
            backend.begin_G(G)
            for record in records:
                path = directory / 'graphs' / ('G' + str(G)) / ('{:04d}.json'.format(record['input_idx']))
                def persist(event, graph, path=path):
                    write(path, graph.snapshot(event), replace=True)
                active = MolecularGraph(record, G, backend, bridge, persist)
                active.run()
                status['completed_graphs'].append({'G': G, 'input_uid': record['input_uid'], 'input_idx': record['input_idx'],
                                                    'graph': str(path.relative_to(ROOT))})
                write(directory / 'unit_status.json', status, replace=True)
                active = None
        assert len(status['completed_graphs']) == 3 * u['n_inputs']
        assert backend.cost['calls_started'] == backend.cost['calls_completed'] == u['planned_generate_calls']
        status['status'] = 'complete'
    except BaseException as err:
        status.update(status='partial', exception=repr(err), traceback=traceback.format_exc())
        if active is not None:
            active.persist('partial_failure')
        raise
    finally:
        close_errors = []
        for component in [bridge, backend]:
            if component is not None:
                try:
                    component.close()
                except BaseException as err:
                    close_errors.append(repr(err))
        status.update(end_utc=utc(), wall_seconds=time.perf_counter() - start, close_errors=close_errors)
        if close_errors:
            status['status'] = 'partial'
        write(directory / 'unit_status.json', status, replace=True)
        if close_errors and status['exception'] is None:
            raise RuntimeError('Resource shutdown failed: ' + repr(close_errors))


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]))
