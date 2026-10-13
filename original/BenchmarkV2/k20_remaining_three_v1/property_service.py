"""Persistent CPU processes; unchanged predictors, rounding and evaluator bodies."""
import importlib.util
import time
import traceback
from common import *

MODULES = {}
MODEL = []
MODEL_LOADS = [0]


def module(name):
    if name not in MODULES:
        spec = importlib.util.spec_from_file_location('frozen_' + name, str(LEGACY / (name + '.py')))
        obj = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(obj)
        obj.write = save
        obj.require_allocation = require_allocation
        MODULES[name] = obj
    return MODULES[name]


def enable_persistent_admet():
    if 'admet_cache_enabled' in MODULES:
        return
    import admet_ai
    constructor = admet_ai.ADMETModel

    def cached_constructor(**kwargs):
        signature = repr(sorted(kwargs.items()))
        if not MODEL:
            MODEL.append((signature, constructor(**kwargs)))
            MODEL_LOADS[0] += 1
        assert MODEL[0][0] == signature, 'Predictor parameters changed'
        return MODEL[0][1]

    admet_ai.ADMETModel = cached_constructor
    MODULES['admet_cache_enabled'] = True


def run_inf(command):
    request = destination(command['requests'])
    output = destination(command['output'])
    data = read(request)
    required = data['method'] != 's1' or data['task_id'] != 'DPQ'
    name = 'admet_worker' if required else 'rdkit_recheck_worker'
    if required:
        enable_persistent_admet()
    obj = module(name)
    old_argv = sys.argv
    loads_before = MODEL_LOADS[0]
    try:
        sys.argv = [name, '--requests', str(request), '--output', str(output)]
        # Correct model-load metadata at publication; no property values are changed.
        def write(path, value):
            if Path(path).name == 'admet_result.json':
                value['identity']['model_load_calls'] = MODEL_LOADS[0] - loads_before
                value['identity']['persistent_model_cache'] = True
                value['identity']['model_cache_hit'] = bool(MODEL) and MODEL_LOADS[0] == loads_before
            save(path, value)
        obj.write = write
        obj.main()
    finally:
        sys.argv = old_argv
    return dict(result_path=str(output / ('admet_result.json' if required else 'chemistry_result.json')),
                actual_model_load_calls=MODEL_LOADS[0] - loads_before)


def run_standard(command):
    from property_bridge import backfill
    from metrics_entry import evaluate_registered, aggregate_registered
    from reflexion import feedback_from_filled
    request = destination(command['requests'])
    data = read(request)
    output = destination(command['output'])
    output.mkdir(exist_ok=False)
    admet = read(destination(command['admet']))
    assert admet['requests_sha256'] == sha(request)
    producer = module('standard_producer')
    producer.produce(data, admet, output)
    responses = read(output / 'responses_raw.json')
    filled = backfill(data['candidates'], data['requests'], data['mapping'], responses, data['cache_contract'])
    byuid = {r['input_uid']: r for r in data['records']}
    individual = []
    for c in filled:
        record = byuid[c['input_uid']]
        native = dict(c, generated_slot_id=c['original_slot_id'], original_slot_id=0, visible_slot_id=0)
        row = evaluate_registered(record['task_id'], record['input_uid'], record['source_properties'], [native], 1)
        for suffix in ('dir', 'delta'):
            row['selected_generated_slot_id_' + suffix] = c['original_slot_id'] if row['SR_' + suffix + '_success'] else None
        feedback = feedback_from_filled(record, c, row)
        individual.append(dict(candidate=next(x for x in data['candidates'] if x['candidate_id'] == c['candidate_id']),
                               outcome=dict(feedback=feedback, filled_candidate=c, metric_row=row)))
    result = dict(filled=filled, individual=individual, requests_sha256=sha(request))
    if data['kind'] == 'terminals':
        rows = []
        for record in data['records']:
            pool = [c for c in filled if c['input_uid'] == record['input_uid']]
            assert len(pool) == 20
            rows.append(evaluate_registered(record['task_id'], record['input_uid'], record['source_properties'], pool, 20))
        result.update(rows=rows, metrics=aggregate_registered(rows))
    save(output / 'result.json.gz', result)
    return dict(result_path=str(output / 'result.json.gz'))


def main():
    require_allocation()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    mode = sys.argv[1]
    assert mode in ('inf', 'standard')
    for line in sys.stdin:
        command = json.loads(line)
        done = destination(command['done'])
        started = time.time()
        try:
            result = (run_inf if mode == 'inf' else run_standard)(command)
            save(done, dict(status='complete', result=result, wall_seconds=time.time() - started))
        except BaseException:
            save(done, dict(status='error', error=traceback.format_exc(), wall_seconds=time.time() - started))
            raise


if __name__ == '__main__':
    main()
