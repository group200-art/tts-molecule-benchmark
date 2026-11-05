"""Continue the exact frozen Mistral R20 experiment in an isolated IO root."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
OLD = REPO / 'BenchmarkV2/k20_remaining_three_v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def bind():
    sys.path.insert(0, str(OLD))
    c = importlib.import_module('common')
    assert c.ROOT in (OLD, ROOT)
    c.ROOT = ROOT
    c.plan.cache_clear()
    return c


def verify_freeze():
    frozen = json.loads((ROOT / 'continuation_freeze_v2.json').read_text())
    for path, expected in frozen['files'].items():
        assert sha(path) == expected, 'Frozen continuation changed: ' + path
    for name in ('plan.json', 'units.json', 'inventory.json'):
        assert sha(ROOT / name) == sha(OLD / name)
    gate = json.loads((ROOT.parent / 'recovery_20261027_v1/validation_gate.json').read_text())
    assert gate['status'] == 'PASS_NORMAL_EQUIVALENCE_AND_KNOWN_FAILURE_RECOVERY'
    return frozen
