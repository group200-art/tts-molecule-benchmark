"""Run unchanged property/rounding algorithms within the new authorized IO scope.

Only allocation validation and output destinations are rebound. Prediction,
parsing, precision, molecular properties, and metric functions are unchanged.
"""
import importlib.util
import sys
from common import ROOT, LEGACY, require_allocation, save, sha, read


def main():
    require_allocation()
    role = sys.argv[1]
    assert role in ('admet', 'standard')
    source = LEGACY / ('admet_worker.py' if role == 'admet' else 'eval_worker.py')
    entry, = [e for e in read(ROOT / 'inventory.json') if e['path'] == str(source)]
    assert sha(source) == entry['sha256']  # The parent already verified the full dependency set.
    spec = importlib.util.spec_from_file_location('frozen_original_' + role, str(source))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def local_write(path, value, compact=False):
        save(path, value)
    module.write = local_write
    module.require_allocation = require_allocation
    sys.argv = [str(source)] + sys.argv[2:]
    module.main()


if __name__ == '__main__':
    main()
