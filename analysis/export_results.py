"""Export archived CSVs with readable labels, preserving every numeric cell."""
import argparse
import csv
import io
from pathlib import Path

from names import RESULT_DIRS, column_name, renamed_row

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check exports without writing files')
    args = parser.parse_args()
    count = 0
    destinations = set()
    for source in sorted((ROOT / 'original/results').rglob('*.csv')):
        relative = source.relative_to(ROOT / 'original/results')
        dest = ROOT / 'results' / RESULT_DIRS[relative.parts[0]] / Path(*relative.parts[1:])
        if dest.name == 'table4_difficulty.csv':
            dest = dest.with_name('table4_reference_groups.csv')
        with source.open(newline='') as f:
            original_rows = list(csv.DictReader(f))
        rows = [renamed_row(row) for row in original_rows]
        # Values are strings throughout: no numeric parsing, rounding, or aggregation.
        for old, new in zip(original_rows, rows):
            for key, value in old.items():
                if key not in ('method_id', 'scope', 'readout'):
                    assert new[column_name(key)] == value
        buffer = io.StringIO(newline='')
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
        content = buffer.getvalue().encode('utf-8')
        if args.check:
            assert dest.read_bytes() == content, 'Export differs: ' + str(dest)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)
        destinations.add(dest)
        count += 1
    assert destinations == set((ROOT / 'results').rglob('*.csv')), 'Unexpected exported CSV'
    print('PASS: {} result CSVs; all numeric cells unchanged.'.format(count))


if __name__ == '__main__':
    main()
