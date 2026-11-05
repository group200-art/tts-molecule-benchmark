"""Run existing frozen batches/controller; only the destination changes."""
import argparse
import importlib
import os
import sys
from runtime import ROOT, bind, verify_freeze


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int)
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    freeze = verify_freeze()
    c = bind()
    c.verify()
    w = importlib.import_module('worker')
    for name in ('common', 'worker', 'replay', 'batched_generation', 'property_client'):
        assert sys.modules[name].ROOT == ROOT
    if args.audit_only:
        assert c.plan()['extensions']['reflexion']['main_id'] == 'Reflexion-3trials-sampled-R20'
        print('PASS: frozen source hashes, isolated write root, unchanged controller and batch identities')
        return
    index = args.index if args.index is not None else int(os.environ['SLURM_ARRAY_TASK_ID'])
    assert index in freeze['unit_indices']
    unit = c.units()[index]
    directory = ROOT / 'runs' / unit['unit_id']
    if args.probe:
        assert index in freeze['probe_indices']
        path = directory / 'probe.json.gz'
        if path.exists():
            value = c.read(path)
            assert value['status'] == 'complete' and value['inputs'] == 8
            assert value['plan_sha256'] == freeze['original_plan_sha256']
            print('Probe already complete:', unit['unit_id'])
            return
    else:
        gate = c.read(ROOT / 'probe_gate.json')
        assert gate['status'] == 'PASS' and gate['continuation_freeze_sha256'] == c.sha(ROOT / 'continuation_freeze.json')
    sys.argv = ['frozen-worker', '--index', str(index)]
    if args.probe:
        # First batch is an exact imported pilot; second batch measures a new
        # frozen batch on the actual allocated device. No output-based selection.
        sys.argv += ['--max-batches', '2']
    w.main()


if __name__ == '__main__':
    main()
