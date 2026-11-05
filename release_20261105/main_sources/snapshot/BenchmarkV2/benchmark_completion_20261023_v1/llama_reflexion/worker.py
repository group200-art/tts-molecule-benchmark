"""Bind the unchanged resumable worker to native Llama sources and isolated IO."""
import argparse
from common import *


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--index', type=int)
    parser.add_argument('--max-batches', type=int)
    args, _ = parser.parse_known_args()
    if args.max_batches:
        require_allocation()
        verify()
        index = args.index if args.index is not None else int(os.environ['SLURM_ARRAY_TASK_ID'])
        unit = units()[index]
        directory = ROOT/'runs'/unit['unit_id']
        path = directory/'probe.json.gz'
        if path.exists():
            data = read(path)
            selected = list(batches(unit))[:args.max_batches]
            inputs = [record for batch in selected for record in batch['records']]
            assert data['status'] == 'complete' and data['probe'] is True and data['unit'] == unit
            assert data['plan_sha256'] == sha(ROOT/'plan.json')
            assert data['batches'] == len(selected) and data['inputs'] == len(inputs)
            assert data['slots'] == 20*len(inputs)
            assert [row['input_uid'] for row in data['rows']] == [record['input_uid'] for record in inputs]
            rows = []
            for batch in selected:
                generation = directory/batch['batch_id']/'generation.json.gz'
                generated = read(generation)
                assert generated['status'] == 'complete' and generated['batch'] == batch
                assert generated['plan_sha256'] == data['plan_sha256']
                terminal = read(directory/batch['batch_id']/'terminals/complete.json')
                assert terminal['generation_sha256'] == sha(generation)
                rows.extend(read(terminal['result_path'])['rows'])
            assert data['rows'] == rows
            print('Verified completed probe; reuse unchanged:', unit['unit_id'], flush=True)
            return
    load_module('frozen_reflexion_worker', ENGINE/'worker.py').main()


if __name__ == '__main__':
    main()
