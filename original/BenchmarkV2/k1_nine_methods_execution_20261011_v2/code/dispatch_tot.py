"""Exactly two GPUs at array level; fixed per-input generation, independent of shards."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['pilot', 'full'])
    args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID'), 'GPU work requires a scheduler allocation'
    index = int(os.environ['SLURM_ARRAY_TASK_ID'])
    if args.stage == 'pilot':
        assert index in (0, 1)
        model = ['mistral', 'llama'][index]
        # A fixed first production shard, used only to validate execution and throughput.
        # The same frozen per-input outputs are counted once when full workers resume.
        units = [dict(model=model, task_id=t, start=0, end=2) for t in ('BDP', 'BHMQ')]
    else:
        unit = json.loads((ROOT / 'config/tot_full_units.json').read_text())[index]
        assert unit['unit_index'] == index
        units = [unit]
    for unit in units:
        cmd = [sys.executable, '-B', str(ROOT / 'code/tot_worker.py'),
               '--model', unit['model'], '--task', unit['task_id'],
               '--start', str(unit['start']), '--end', str(unit['end']), '--stage', args.stage]
        print(json.dumps({'event': 'dispatch', 'command': cmd}), flush=True)
        subprocess.run(cmd, check=True, cwd=str(ROOT))


if __name__ == '__main__':
    main()
