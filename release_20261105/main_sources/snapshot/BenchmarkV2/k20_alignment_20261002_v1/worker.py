"""Generate exactly twenty terminal outputs per input, including failures."""
import os
import sys
import time
import traceback
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from contracts import ROOT, APPROVAL, read, write, sha, digest, pilot_config, manifest, repeat_seed
from runtime_common import ENVS, require_allocation, isolated_env, checked_child, verify_code_inventory
from local_properties import LocalProperties


def terminal_candidate(unit, record, repeat, seed, text, smiles):
    identity = dict(run_id=APPROVAL, model_id='gellmo_p6_mistral', method_id=unit['method_id'],
                    task_id=record['task_id'], input_uid=record['input_uid'], original_slot_id=repeat)
    return dict(identity, candidate_id='k20-' + digest(identity), input_idx=record['input_idx'],
                source_smiles=record['source_smiles'], repeat_id=repeat, seed=seed, raw_text=text, raw_smiles=smiles)


def run_record(unit, record, gen, props, directory, costs):
    """A fresh controller/episodic memory for every repeat, even after success."""
    method = unit['method_id']
    slots = []
    if method == 'fewshot':
        plans = read(ROOT / ('configs/pplus_prompts_' + record['task_id'] + '.json'))
        plan = next(p for p in plans if p['record']['input_uid'] == record['input_uid'])
        assert plan['record'] == record and plan['shots'] == 4
    for repeat in range(20):
        seed = repeat_seed(method, record['task_id'], record['input_uid'], repeat)
        gen.set_seed(seed)
        folder = directory / 'trajectories' / ('input{:04d}_repeat{:02d}'.format(record['input_idx'], repeat))
        folder.mkdir(parents=True, exist_ok=False)
        context = dict(task_id=record['task_id'], input_uid=record['input_uid'], input_idx=record['input_idx'], repeat_id=repeat, seed=seed)
        if method == 's1':
            from p9_adapters import build_prompt, parse_final
            from s1bf import run_one_budget
            prompt = build_prompt(record)
            ids = gen.tok.encode(prompt, add_special_tokens=True, truncation=False)
            trace = run_one_budget(gen, ids, 512, context)
            text = trace['final_text']
            candidate = terminal_candidate(unit, record, repeat, seed, text, parse_final(text)['raw_smiles'])
        elif method == 'fewshot':
            from pplus_parser import parse_smiles
            output = gen.generate(plan['prompt'], context)[0]
            parsed = parse_smiles(output['raw_text'])
            candidate = terminal_candidate(unit, record, repeat, seed, output['raw_text'], parsed[0])
            trace = dict(prompt_sha256=plan['prompt_sha256'], demo_uids=plan['demo_uids'], output=output, parser=list(parsed))
        elif method == 'reflexion':
            from reflexion import run_input
            templates = read(ROOT / 'configs/reflexion_prompts.json')

            def emit(trial, name, value):
                write(folder / ('trial{:02d}_{}.json'.format(trial, name)), value)

            def call(role, prompt, ctx):
                return dict(gen.generate(prompt, dict(ctx, repeat_id=repeat, seed=seed))[0], role=role)

            def evaluate(item):
                bridge = props.bridge([item])
                trial = item['trial_index']
                inp = folder / ('trial{:02d}_eval_input.json'.format(trial))
                out = folder / ('trial{:02d}_eval_result.json'.format(trial))
                write(inp, dict(record=record, candidate=item, bridge=bridge))
                checked_child(ENVS['gellmo_eval_py36'], 'eval_worker.py',
                              ['--mode', 'evaluate_trial', '--input', inp, '--input-sha', sha(inp), '--output', out],
                              folder, 'evaluate_trial' + str(trial), costs['property_processes'])
                return read(out)

            # Different run identity prevents candidate-id collisions across repeats.
            trace = run_input(record, APPROVAL + '-repeat' + str(repeat), templates, call, evaluate, emit)
            old = trace['native_candidate']
            candidate = terminal_candidate(unit, record, repeat, seed, old['raw_text'], old['raw_smiles'])
            candidate['terminal_trial'] = trace['terminal_trial']
            candidate['trajectory_candidate_id'] = old['candidate_id']
        else:
            raise ValueError(method)
        write(folder / 'complete.json', dict(context=context, trajectory=trace, terminal_candidate=candidate))
        slots.append(candidate)
        costs['terminal_candidates_complete'] += 1
    assert len(slots) == 20 and [c['original_slot_id'] for c in slots] == list(range(20))
    assert len({c['candidate_id'] for c in slots}) == 20
    return slots


def main():
    unit = require_allocation()
    verify_code_inventory()
    directory = ROOT / 'runs' / unit['unit_id']
    directory.mkdir(parents=True, exist_ok=False)
    os.environ.update(isolated_env(directory))
    os.chdir(str(directory))
    records = manifest(unit['task_id'], unit['unit_id'])
    write(directory / 'inputs.json', records)
    costs = dict(load_started=0, load_completed=0, generate_started=0, generate_completed=0,
                 generate_cuda_wall_seconds=0., physical_return_sequences=0, prompt_tokens_unexpanded=0,
                 returned_tokens_through_first_eos=0, terminal_candidates_complete=0,
                 property_processes=[], property_events=[])
    status = dict(status='started', unit=unit, protocol_sha256=sha(ROOT / 'configs/protocol.json'))
    write(directory / 'started.json', status)
    start = time.perf_counter()
    try:
        from generation import Generation
        gen = Generation(unit['method_id'], pilot_config()['methods'][unit['method_id']], costs,
                         directory, unit['maximum_generate_calls'])
        gen.start()
        props = LocalProperties(unit['task_id'], directory, costs, unit['method_id'])
        candidates = []
        for record in records:
            candidates.extend(run_record(unit, record, gen, props, directory, costs))
            print('Completed 20 outputs:', unit['method_id'], unit['task_id'], record['input_idx'], flush=True)
        write(directory / 'terminal_candidates.json', candidates)
        inp = directory / 'native_bundle.json'
        write(inp, dict(unit_id=unit['unit_id'], model_id='gellmo_p6_mistral', records=records,
                        candidates=candidates, bridge=props.bridge(candidates)))
        checked_child(ENVS['gellmo_eval_py36'], 'eval_worker.py',
                      ['--mode', 'native', '--input', inp, '--input-sha', sha(inp), '--output', directory / 'metrics.json'],
                      directory, 'native_metrics', costs['property_processes'])
        assert costs['terminal_candidates_complete'] == len(records) * 20
        assert costs['generate_started'] == costs['generate_completed'] <= unit['maximum_generate_calls']
        assert read(directory / 'metrics.json')['status'] == 'complete'
        status['status'] = 'complete'
    except BaseException as error:
        status.update(status='failed', exception=repr(error))
        write(directory / 'exception.json', dict(exception=repr(error), traceback=traceback.format_exc()))
        raise
    finally:
        costs['worker_wall_seconds'] = time.perf_counter() - start
        write(directory / 'costs.json', costs)
        write(directory / 'status.json', status)


if __name__ == '__main__':
    main()
