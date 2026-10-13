"""Read-only local progress; does not query Slurm or infer unfinished scores."""
from common import *


def main():
    report = dict(method=METHOD, required_new_trajectories=66385, existing_reused=11715,
                  generated=0, evaluated=0, generation_complete_seeds=0, evaluation_complete_seeds=0, seeds=[])
    for seed in range(51004, 51021):
        row = dict(seed=seed, generated=0, evaluated=0)
        for name, field in [('generation', 'generated'), ('evaluation', 'evaluated')]:
            d = ROOT/name/str(seed)
            complete = d/'complete.json'
            progress = d/'progress.json'
            if complete.exists():
                row[field] = 3905
                report[name+'_complete_seeds'] += 1
            elif progress.exists():
                x = read(progress)
                row[field] = x.get('completed_trajectories', x.get('completed_candidates', 0))
        report['generated'] += row['generated']
        report['evaluated'] += row['evaluated']
        report['seeds'].append(row)
    for name in ('probe_gate.json', 'production_submission.json', 'results/summary.json'):
        if (ROOT/name).exists():
            report[name] = read(ROOT/name)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
