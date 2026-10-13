"""Read checkpoint completion; never label partial scores as full benchmark results."""
from common import *


def main():
    result = {}
    for method in METHODS:
        done, generated, evaluated = 0, 0, 0
        for unit in units():
            if unit['method'] != method:
                continue
            directory = ROOT / 'runs' / unit['unit_id']
            done += (directory / 'complete.json').exists()
            for batch in batches(unit):
                n = len(batch['lanes'])
                generated += n if (directory / batch['batch_id'] / 'generation.json.gz').exists() else 0
                evaluated += n if (directory / batch['batch_id'] / 'terminals/complete.json').exists() else 0
        result[method] = dict(full_units=done, total_units=56, generated_terminal_slots=generated,
                             evaluated_terminal_slots=evaluated, required_slots=78100)
    result['nine_method_summary_ready'] = (ROOT / 'results/summary.json').exists()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
