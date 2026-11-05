"""Expand the one-copy frozen ICL prompt bundle into a separate staging tree."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'SOURCE_MANIFEST.json').read_text())
    count = 0
    for row in manifest['files']:
        if not row['role'].startswith('frozen_input_specific_four_shot_prompts'):
            continue
        compressed = (root / row['path']).read_bytes()
        assert hashlib.sha256(compressed).hexdigest() == row['released_sha256']
        data = gzip.decompress(compressed)
        assert hashlib.sha256(data).hexdigest() == row['uncompressed_released_sha256']
        for target in [row['original_repository_path']] + row['identical_original_copies']:
            path = args.destination / target
            if path.exists():
                assert path.read_bytes() == data, 'Refusing to replace a different file: ' + str(path)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as stream:
                    stream.write(data)
            count += 1
    print(json.dumps(dict(status='PASS', materialized_files=count, generation_calls=0)))


if __name__ == '__main__':
    main()
