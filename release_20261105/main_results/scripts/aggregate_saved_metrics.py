"""Saved-readout aggregation extracted unchanged from finalize_benchmark.py.

The body of aggregate and mean are byte-for-byte source excerpts. Candidate
eligibility, representative selection and property prediction are not rerun.
"""
import math

def mean(values):
    values = list(values)
    return math.fsum(values) / len(values) if values and all(v is not None for v in values) else None

TASKS = ['BDP', 'BDQ', 'BPQ', 'DPQ', 'BDPQ', 'MPQ', 'BDMQ', 'BHMQ', 'BMPQ', 'HMPQ']
METHODS = ['beam', 'sampling', 'self_refine', 'tot', 'got', 's1', 'voting', 'icl', 'reflexion']
MODELS = ['Mistral', 'Llama']
KS = [1, 5, 10, 20]


def aggregate(groups, expected):
    compact, per_task, macro, prefixes = [], [], [], []
    for model in MODELS:
        for method in METHODS:
            inputs = groups[model, method]
            assert set(inputs) == set(expected)
            task_rows = []
            for task in TASKS:
                rows = [inputs[key] for key in expected if key[0] == task]
                good = [row for row in rows if row['success']]
                result = dict(model=model, method_id=method, task_id=task, n_inputs=len(rows),
                              n_success=len(good), n_delta=sum(row['delta'] for row in rows),
                              n_valid_slots=sum(row['n_valid_slots'] for row in rows),
                              SR20=100 * len(good) / len(rows),
                              Valid_slots=100 * sum(row['n_valid_slots'] for row in rows) / (20 * len(rows)),
                              SR_delta20=100 * sum(row['delta'] for row in rows) / len(rows),
                              **{key: mean(row[key] for row in good) for key in ('RI_01', 'Sim', 'SAS')})
                per_task.append(result)
                task_rows.append(result)
                for k in KS:
                    n_success = sum(row['SR_K' + str(k)] for row in rows)
                    prefixes.append(dict(model=model, method_id=method, task_id=task, K=k,
                                         n_inputs=len(rows), n_success=n_success, SR=100 * n_success / len(rows)))
                for row in rows:
                    assert all(not row['SR_K' + str(a)] or row['SR_K' + str(b)] for a, b in zip(KS, KS[1:]))
                    compact.append(dict(model=model, method_id=method, **row))
            macro.append(dict(model=model, method_id=method, n_inputs=3905, n_slots=78100,
                              n_success=sum(row['n_success'] for row in task_rows),
                              IND=mean(row['SR20'] for row in task_rows[:5]),
                              OOD=mean(row['SR20'] for row in task_rows[5:]),
                              **{key: mean(row[key] for row in task_rows)
                                 for key in ('SR20', 'Valid_slots', 'SR_delta20', 'RI_01', 'Sim', 'SAS')},
                              tasks_with_success=sum(row['n_success'] > 0 for row in task_rows)))
    assert len(compact) == 70290 and len(per_task) == 180 and len(macro) == 18
    return per_task, macro, prefixes
