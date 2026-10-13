"""Verify the included experiment originals without the cluster filesystem."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    records = json.loads((ROOT / 'evidence/source_manifest.json').read_text())
    for r in records:
        path = ROOT / r['destination']
        blob = path.read_bytes()
        if len(blob) != r['bytes'] or hashlib.sha256(blob).hexdigest() != r['sha256']:
            raise SystemExit('Hash mismatch: ' + r['destination'])
    print('PASS: {} experiment source files match the snapshot.'.format(len(records)))


if __name__ == '__main__':
    main()
