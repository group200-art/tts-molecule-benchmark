"""Verify this source release without loading models, chemistry, or run modules."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / 'SOURCE_MANIFEST.json').read_text())
    files = manifest['files']
    assert len({r['path'] for r in files}) == len(files), 'Duplicate released path'
    omitted = manifest.get('omitted_execution_assets', [])
    for item in omitted:
        path = root / item.get('release_path', 'snapshot/' + item['original_repository_path'])
        assert not path.exists(), 'Excluded execution archive is present: ' + str(path)
    assert any(item.get('category') == 'operational_recovery_archive' for item in omitted)
    count_python = count_json = count_prompts = 0
    for item in files:
        path = (root / item['path']).resolve()
        assert root in path.parents, 'Path leaves component root'
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == item['released_sha256'], item['path']
        assert len(data) == item['released_bytes'], item['path']
        if item.get('storage') == 'gzip':
            data = gzip.decompress(data)
            assert hashlib.sha256(data).hexdigest() == item['uncompressed_released_sha256']
        if path.suffix == '.py':
            compile(data, str(path), 'exec')
            count_python += 1
        if path.suffix == '.json' or item.get('storage') == 'gzip':
            value = json.loads(data)
            count_json += 1
            if item['role'].startswith('frozen_input_specific_four_shot_prompts'):
                for row in value:
                    assert row['shots'] == 4 and len(row['demo_uids']) == 4
                    assert hashlib.sha256(row['prompt'].encode()).hexdigest() == row['prompt_sha256']
                count_prompts += len(value)
    assert count_prompts == 3905, count_prompts
    configurations = json.loads((root / 'MAIN_CONFIGURATIONS.json').read_text())['configurations']
    assert len(configurations) == 18
    assert len({(r['backbone'], r['method']) for r in configurations}) == 18
    for row in configurations:
        assert (root / row['entrypoint']).is_file(), row['entrypoint']
        for path in row['contracts']:
            assert (root / path).is_file(), path
    print(json.dumps(dict(status='PASS', files=len(files), python_compiled=count_python,
        json_parsed=count_json, frozen_icl_prompts=count_prompts, configurations=18,
        omitted_execution_assets=len(omitted),
        model_calls=0, property_prediction_calls=0), indent=2))


if __name__ == '__main__':
    main()
