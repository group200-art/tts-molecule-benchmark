"""Read-only, cached instances of the accepted registered metric functions.

No chemistry/model imports, prediction, precision conversion, or old-directory writes.
The function bodies are executed verbatim from their checksum-protected source.
"""
import ast
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
B = ROOT.parent
REPO = B.parent
LEGACY = B / 'next3_ten_tasks_run01_v1'


def load_metrics():
    read_set = json.loads((LEGACY / 'evidence/read_set.json').read_text())
    expected = {r['path']: r['sha256'] for r in read_set}
    paths = [B / 'src/metrics_cpu.py', B / 'registries/task_registry.draft.json']
    for path in paths:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected[str(path)]
    kernel = {}
    exec(compile(paths[0].read_text(), str(paths[0]), 'exec'), kernel)
    tasks = {t['task_id']: t for t in json.loads(paths[1].read_text())['tasks']}

    def task_contract(task_id, supplied=None):
        task = tasks[task_id]
        if supplied is not None and supplied != task['objectives']:
            raise ValueError('Task objective drift')
        return task

    def finite_or_none(value):
        try:
            x = float(value)
            return x if not isinstance(value, bool) and math.isfinite(x) else None
        except (TypeError, ValueError, OverflowError):
            return None

    source = LEGACY / 'src/metrics_entry.py'
    tree = ast.parse(source.read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('evaluate_registered', 'aggregate_registered')]
    assert len(tree.body) == 2
    namespace = {'kernel': lambda: kernel, 'task_contract': task_contract,
                 'finite_or_none': finite_or_none}
    exec(compile(ast.fix_missing_locations(tree), str(source), 'exec'), namespace)
    return namespace['evaluate_registered'], namespace['aggregate_registered'], kernel, tasks


evaluate_registered, aggregate_registered, KERNEL, TASK_REGISTRY = load_metrics()
