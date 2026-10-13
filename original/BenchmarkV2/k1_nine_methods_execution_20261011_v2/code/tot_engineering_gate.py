"""Fail-closed pilot engineering audit; never gate on molecular/judge quality."""
import argparse
import json
import math
import statistics
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from tot_protocol import (ROOT, BV2, MODEL_IDS, TASKS, read, save, sha, digest, manifest, task_spec,
                          initial_prompt, child_prompt, judge_prompt, parse_smiles, vote_ranking,
                          batch_seed, validate_freeze)

PILOT_KEYS = tuple((model, task, idx) for model in MODEL_IDS for task in ('BDP', 'BHMQ') for idx in (0, 1))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_nonnegative(value, label):
    require(isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0,
            'Invalid nonnegative numeric field: '+label)


def validate_units(units, task_records):
    require(isinstance(units, list) and len(units) == 80, 'Full plan must have exactly 80 units')
    expected = {(model, task, r['input_idx']): r['input_uid']
                for model in MODEL_IDS for task, records in task_records.items() for r in records}
    require(len(expected) == 7810, 'Full source manifest must contain 7810 model-input pairs')
    actual = {}
    for index, unit in enumerate(units):
        require(unit['unit_index'] == index, 'Unit index/order mismatch')
        model, task = unit['model'], unit['task_id']
        require(model in MODEL_IDS and task in TASKS, 'Unknown model/task in full plan')
        start, end = unit['start'], unit['end']
        require(type(start) is int and type(end) is int and 0 <= start < end <= len(task_records[task]),
                'Full unit input bounds mismatch')
        records = task_records[task][start:end]
        require(unit['input_uids'] == [r['input_uid'] for r in records], 'Full unit UID/order mismatch')
        for record in records:
            key = (model, task, record['input_idx'])
            require(key not in actual, 'Overlapping full units: '+str(key))
            actual[key] = record['input_uid']
    require(actual == expected, 'Full plan omission/addition/identity mismatch')
    require(all(key in actual for key in PILOT_KEYS), 'Pilot keys not reusable in full plan')
    return {'units': len(units), 'model_inputs': len(actual), 'overlaps': 0,
            'missing': 0, 'pilot_endpoints_reused_once': 8, 'new_remaining_inputs': 7802}


def expected_config(contract, model, phase):
    from transformers import GenerationConfig
    spec = contract['tokenizer'][model]
    cap = contract['child_max_new_tokens'] if phase == 'children' else contract['judge_max_new_tokens']
    config = GenerationConfig.from_dict(read(contract['generation_defaults_path']))
    require(not config.update(**dict(contract['sampling'], max_new_tokens=cap,
        bos_token_id=spec['bos'], eos_token_id=spec['eos'], pad_token_id=spec['pad'])), 'Unused decoding keys')
    config.validate()
    return config.to_dict()


def validate_call(call, model, task, idx, phase, prompts, path, contract, contract_sha, tokenizer,
                  effective_config=None):
    require(call['status'] == 'complete', 'Incomplete physical batch')
    seed = batch_seed(model, task, idx, phase)
    count = 15 if phase == 'children' else 5
    cap = contract['child_max_new_tokens'] if phase == 'children' else contract['judge_max_new_tokens']
    prompt_limit = contract['child_prompt_limit'] if phase == 'children' else contract['judge_prompt_limit']
    token_spec = contract['tokenizer'][model]
    inputs = [tokenizer.encode(p, add_special_tokens=True, truncation=False) for p in prompts]
    expected_request = {'model': model, 'contract_sha256': contract_sha, 'phase': phase,
                        'seed': seed, 'prompts': prompts, 'input_ids': inputs, 'max_new_tokens': cap}
    require(call['request'] == expected_request, 'Call prompt/input/seed/contract mismatch: '+str(path))
    require(len(prompts) == count and len(call['outputs']) == count, 'Physical batch response count mismatch')
    for ids in inputs:
        require(ids and ids[0] == token_spec['bos'] and len(ids) <= prompt_limit and
                len(ids)+cap <= contract['context_limit'], 'Prompt prefix/context bounds mismatch')
    expected = expected_config(contract, model, phase) if effective_config is None else effective_config
    require(call['effective_decoding'] == expected, 'Effective decoding changed')
    finite_nonnegative(call['physical_cuda_wall_seconds'], 'batch elapsed')
    require(call['physical_cuda_wall_seconds'] > 0, 'Missing measured GPU time')
    finite_nonnegative(call['peak_allocated_bytes'], 'peak allocated bytes')
    require(bool(call['slurm_job_id']), 'Missing allocation identity')
    padded_lengths = []
    for row, output in enumerate(call['outputs']):
        require(output['batch_row'] == row and output['seed'] == seed and output['call_path'] == str(path),
                'Output batch identity mismatch')
        generated, content = output['generated_ids'], output['content_ids']
        require(isinstance(generated, list) and 1 <= len(generated) <= cap and
                all(type(v) is int and v >= 0 for v in generated), 'Generated token count/type mismatch')
        eos = token_spec['eos']
        ended = generated[-1] == eos
        require(eos not in generated[:-1], 'Tokens retained after first EOS')
        require(output['ended_with_eos'] == ended, 'EOS flag mismatch')
        require(content == (generated[:-1] if ended else generated), 'Content/EOS trimming mismatch')
        require(output['raw_text'] == tokenizer.decode(content, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False), 'Raw response/token decode mismatch')
        require(output['input_tokens'] == len(inputs[row]) and output['output_tokens'] == len(generated),
                'Token accounting mismatch')
        padding = output['post_eos_padding_tokens']
        require(type(padding) is int and 0 <= padding <= cap-len(generated), 'Padding count mismatch')
        require(ended or padding == 0, 'Unfinished output unexpectedly padded')
        require(output['cap_exhausted'] == (len(generated) == cap), 'Cap exhaustion flag mismatch')
        require(output['reached_cap_without_eos'] == (not ended and len(generated) == cap), 'Length-stop flag mismatch')
        padded_lengths.append(len(generated)+padding)
    require(len(set(padded_lengths)) == 1, 'Physical batch rows have unequal padded output length')
    require(padded_lengths[0] == max(len(o['generated_ids']) for o in call['outputs']), 'Extra unexplained batch padding')


def validate_prefix_sources(prefix, model, objectives):
    record = prefix['record']
    provenance = prefix['initial_provenance']
    require(sha(provenance['source_prefix_path']) == provenance['source_prefix_sha256'], 'Legacy initial prefix changed')
    require(read(provenance['source_prefix_path']) == prefix['initial_candidates'], 'Legacy initial candidates changed')
    identity = provenance['model_identity']
    require(sha(identity['path']) == identity['sha256'], 'Legacy model identity changed')
    require(read(identity['path'])['load_spec'] == identity['load_spec'], 'Legacy model checkpoint mismatch')
    require(len(provenance['calls']) == 5, 'Initial logical response count mismatch')
    for i, evidence in enumerate(provenance['calls']):
        require(sha(evidence['path']) == evidence['sha256'], 'Legacy initial generation call changed')
        call = read(evidence['path'])
        require(call['input_ids'] == evidence['input_ids'] and call['seed'] == evidence['seed'], 'Legacy input/seed changed')
        require(call['prompt'] == evidence['prompt'] == initial_prompt(record, objectives, i), 'Legacy initial prompt changed')
        candidate = prefix['initial_candidates'][i]
        require(candidate['input_uid'] == record['input_uid'] and candidate['model_id'] == MODEL_IDS[model] and
                candidate['input_idx'] == record['input_idx'] and candidate['original_slot_id'] == i and
                candidate['stage'] == 'depth1', 'Reused candidate identity mismatch')
        raw = call['outputs'][0]['decoded_continuation_untrimmed' if model == 'mistral' else 'raw_text']
        require(raw == candidate['raw_text'] and parse_smiles(raw) == candidate['parse_status'], 'Legacy raw/parser mismatch')


def validate_completed(result, prefix, calls, model, directory, objectives, contract, contract_sha, tokenizer,
                       configs=None):
    record = prefix['record']
    task, idx, uid = record['task_id'], record['input_idx'], record['input_uid']
    require(result['status'] == 'complete_generation_pending_property_evaluation', 'Pilot generation incomplete')
    require(result['schema_version'] == 'BV2-TOT-GENERATION-ENDPOINT-v2', 'Unexpected endpoint schema')
    for key, value in {'model_id': MODEL_IDS[model], 'method_id': 'tot_lm_vote', 'task_id': task,
                       'input_uid': uid, 'input_idx': idx, 'source_smiles': record['source_smiles'],
                       'source_properties': record['source_properties'], 'contract_sha256': contract_sha}.items():
        require(result[key] == value, 'Endpoint identity mismatch: '+key)
    require(result['origin_stage'] == 'pilot', 'Expected unchanged production pilot endpoint')
    require(result['prefix_digest'] == digest(prefix), 'Endpoint prefix identity mismatch')
    require(result['initial_candidates'] == prefix['initial_candidates'], 'First frontier filtered/replaced')
    initial = prefix['initial_candidates']
    parent_prompts = [judge_prompt(record, objectives, initial)]*5
    validate_call(calls['parent_vote'], model, task, idx, 'parent_vote', parent_prompts,
                  directory/'parent_vote.json', contract, contract_sha, tokenizer,
                  None if configs is None else configs['parent_vote'])
    pv = vote_ranking(calls['parent_vote']['outputs'], 5)
    require(result['parent_vote'] == pv and result['selected_parent_indices'] == pv['ranking'][:3],
            'Parent selection does not follow actual LM votes')
    parents = [initial[i] for i in pv['ranking'][:3]]
    child_prompts = [child_prompt(record, objectives, p) for p in parents for _ in range(5)]
    validate_call(calls['children'], model, task, idx, 'children', child_prompts,
                  directory/'children.json', contract, contract_sha, tokenizer,
                  None if configs is None else configs['children'])
    children = result['children']
    require(len(children) == 15, 'Last frontier filtered/replaced')
    for i, (child, output) in enumerate(zip(children, calls['children']['outputs'])):
        parent = parents[i//5]
        identity = {'model_id': MODEL_IDS[model], 'method_id': 'tot_lm_vote', 'task_id': task,
                    'input_uid': uid, 'stage': 'depth2', 'node': f'p{i//5+1}:b{i%5}',
                    'original_slot_id': 5+i, 'contract_sha256': contract_sha}
        parsed = parse_smiles(output['raw_text'])
        expected = dict(identity, candidate_id='candidate-v2-'+digest(identity), input_idx=idx,
            raw_text=output['raw_text'], raw_smiles=parsed['parsed_smiles'], parse_status=parsed,
            source_smiles=record['source_smiles'], parent_candidate_id=parent['candidate_id'],
            parent_original_slot_id=parent['original_slot_id'], parent_rank=i//5+1, branch=i%5,
            call_path=str(directory/'children.json'), batch_row=i, prompt_sha256=digest(child_prompts[i]),
            generation_seed=output['seed'])
        require(child == expected, 'Child causal identity/raw/provenance mismatch: '+str(i))
    terminal_prompts = [judge_prompt(record, objectives, children)]*5
    validate_call(calls['terminal_vote'], model, task, idx, 'terminal_vote', terminal_prompts,
                  directory/'terminal_vote.json', contract, contract_sha, tokenizer,
                  None if configs is None else configs['terminal_vote'])
    tv = vote_ranking(calls['terminal_vote']['outputs'], 15)
    require(result['terminal_vote'] == tv, 'Terminal vote count/parse mismatch')
    chosen = children[tv['ranking'][0]]
    expected_terminal = dict(chosen, source_original_slot_id=chosen['original_slot_id'], original_slot_id=0)
    require(result['selected_candidate'] == expected_terminal, 'Terminal output not actual LM-selected last-frontier child')
    provenance = result['selection_provenance']
    require(provenance['source_path'] == str(directory/'complete.json') and provenance['contract_sha256'] == contract_sha,
            'Endpoint source/contract provenance mismatch')
    require(provenance['selected_terminal_frontier_index'] == tv['ranking'][0] and
            provenance['selected_candidate_id'] == chosen['candidate_id'], 'Terminal selection provenance mismatch')
    require(provenance['candidate_property_access'] is False and provenance['external_validity_filter'] is False,
            'External evaluation/filter claimed during generation')
    require(provenance['parent_vote_call_path'] == str(directory/'parent_vote.json') and
            provenance['terminal_vote_call_path'] == str(directory/'terminal_vote.json'), 'Vote call provenance mismatch')
    call_order = [calls[p] for p in ('parent_vote', 'children', 'terminal_vote')]
    expected_costs = {'reused_initial_responses': 5, 'new_molecule_responses': 15, 'judge_responses': 10,
        'new_physical_generate_calls': 3, 'logical_model_responses_total': 30,
        'reused_initial_gpu_seconds': sum(c['cuda_wall_seconds'] for c in prefix['initial_provenance']['calls']),
        'reused_initial_prompt_tokens': sum(len(c['input_ids']) for c in prefix['initial_provenance']['calls']),
        'reused_initial_output_tokens': sum(c['output_tokens'] for c in prefix['initial_provenance']['calls']),
        'new_physical_cuda_wall_seconds': sum(c['physical_cuda_wall_seconds'] for c in call_order),
        'new_prompt_tokens_expanded': sum(o['input_tokens'] for c in call_order for o in c['outputs']),
        'new_output_tokens_through_eos': sum(o['output_tokens'] for c in call_order for o in c['outputs']),
        'property_queries_during_generation': 0}
    require(result['costs'] == expected_costs, 'Logical/physical token/time/property-query accounting mismatch')
    return {'model': model, 'task_id': task, 'input_idx': idx, 'input_uid': uid,
            'generation_seconds': expected_costs['new_physical_cuda_wall_seconds'],
            'phase_seconds': {p: calls[p]['physical_cuda_wall_seconds'] for p in calls},
            'parent_valid_votes': pv['valid_votes'], 'terminal_valid_votes': tv['valid_votes'],
            'parent_all_invalid_fallback': pv['all_invalid_fallback'], 'terminal_all_invalid_fallback': tv['all_invalid_fallback'],
            'child_empty_parses': sum(not c['raw_smiles'] for c in children),
            'judge_responses_at_cap_without_eos': sum(o['reached_cap_without_eos'] for p in ('parent_vote', 'terminal_vote') for o in calls[p]['outputs']),
            'maximum_prompt_tokens': max(o['input_tokens'] for c in call_order for o in c['outputs']),
            'quality_affects_gate': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'reports/tot_engineering_gate.json')
    args = parser.parse_args()
    started = time.time()
    evidence = {}
    try:
        contract, contract_sha = validate_freeze()
        tasks = {task: manifest(task) for task in TASKS}
        unit_path = ROOT/'config/tot_full_units.json'
        units = read(unit_path)
        coverage = validate_units(units, tasks)
        evidence[str(unit_path)] = sha(unit_path)
        from transformers import AutoTokenizer
        tokenizers = {}
        for model in MODEL_IDS:
            tok = AutoTokenizer.from_pretrained(contract['models'][model]['base_path'],
                use_fast=contract['tokenizer'][model]['use_fast'], local_files_only=True, trust_remote_code=False)
            require(tok.bos_token_id == contract['tokenizer'][model]['bos'] and
                    tok.eos_token_id == contract['tokenizer'][model]['eos'], 'Tokenizer identity mismatch')
            tokenizers[model] = tok
        prefixes = {(m, t): read(ROOT/'config'/f'tot_prefix_{m}_{t}.json') for m in MODEL_IDS for t in ('BDP', 'BHMQ')}
        diagnostics = []
        load_seconds = defaultdict(list)
        model_identity_paths = set()
        for model, task, idx in PILOT_KEYS:
            directory = ROOT/'runs'/f'tot_{model}_{task}'/f'{idx:04d}'
            prefix = read(directory/'initial_prefix.json')
            require(prefix == prefixes[(model, task)][idx], 'Frozen/live initial prefix mismatch')
            require(prefix['record'] == tasks[task][idx], 'Pilot UID/source manifest mismatch')
            objectives = task_spec(task)['objectives']
            validate_prefix_sources(prefix, model, objectives)
            result = read(directory/'complete.json')
            phases = ('parent_vote', 'children', 'terminal_vote')
            calls = {phase: read(directory/f'{phase}.json') for phase in phases}
            diagnostics.append(validate_completed(result, prefix, calls, model, directory, objectives,
                                                  contract, contract_sha, tokenizers[model]))
            for phase, call in calls.items():
                started_call = read(directory/f'{phase}.started.json')
                require(started_call['request'] == call['request'] and started_call['effective_decoding'] == call['effective_decoding'],
                        'Started/completed call causal evidence mismatch')
                identity_path = ROOT/'runs/tot_jobs'/f"{model}_{call['slurm_job_id']}_{task}_0_2"/'model_identity.json'
                identity = read(identity_path)
                require(identity['model_id'] == MODEL_IDS[model] and identity['contract_sha256'] == contract_sha and
                        identity['load_spec'] == contract['models'][model], 'Real actor/judge model differs from contract')
                require(identity['adapter_status']['active_adapters'] == ['default'] and
                        identity['adapter_status']['merged_adapters'] == [] and identity['adapter_status']['enabled'] is True,
                        'Model adapter inactive/merged')
                require(identity['training'] is False, 'Model in training mode')
                if identity_path not in model_identity_paths:
                    model_identity_paths.add(identity_path)
                    load_seconds[model].append(identity['load_seconds'])
                    completion = read(identity_path.parent/'completion.json')
                    require(completion['status'] == 'complete' and completion['stage'] == 'pilot' and
                            completion['start'] == 0 and completion['end'] == 2 and len(completion['inputs']) == 2,
                            'Pilot worker completion/shard receipt mismatch')
                for path in (directory/f'{phase}.json', directory/f'{phase}.started.json', identity_path):
                    evidence[str(path)] = sha(path)
            for path in (directory/'initial_prefix.json', directory/'complete.json'):
                evidence[str(path)] = sha(path)
        by_model = {}
        for model in MODEL_IDS:
            values = [r['generation_seconds'] for r in diagnostics if r['model'] == model]
            require(len(values) == 4, 'Insufficient fixed pilot input coverage')
            load = statistics.mean(load_seconds[model])
            by_model[model] = {'observed_inputs': 4, 'seconds_per_input_mean': statistics.mean(values),
                'seconds_per_input_min': min(values), 'seconds_per_input_max': max(values),
                'seconds_per_model_load_mean': load, 'remaining_inputs': 3901,
                'full_model_seconds_mean_projection': 3901*statistics.mean(values)+40*load,
                'full_model_seconds_min_observed_projection': 3901*min(values)+40*load,
                'full_model_seconds_max_observed_projection': 3901*max(values)+40*load}
        throughput = {'per_model': by_model,
            'two_gpu_ideal_remaining_hours_mean': sum(v['full_model_seconds_mean_projection'] for v in by_model.values())/7200,
            'two_gpu_min_max_observed_scenario_hours': [
                sum(v['full_model_seconds_min_observed_projection'] for v in by_model.values())/7200,
                sum(v['full_model_seconds_max_observed_projection'] for v in by_model.values())/7200],
            'meaning': 'Small pilot extrapolation, not a confidence interval or deadline guarantee; excludes queue time, property evaluation and unmeasured long-tail/I/O costs.'}
        report = {'status': 'PASS_ENGINEERING_ONLY_NOT_METHOD_EFFECTIVENESS', 'contract_sha256': contract_sha,
            'pilot_inputs': 8, 'pilot_physical_batches': 24, 'pilot_new_model_responses': 200,
            'pilot_reused_initial_responses': 40, 'full_coverage': coverage, 'diagnostics': diagnostics,
            'throughput': throughput, 'forbidden_quality_gates': ['SR', 'molecular_validity', 'valid_vote_rate'],
            'external_property_evaluation_performed': False, 'evidence_sha256': evidence,
            'gate_script_sha256': sha(Path(__file__)), 'checked_unix': time.time(), 'wall_seconds': time.time()-started}
        save(args.output, report)
        print(json.dumps({k: v for k, v in report.items() if k not in ('evidence_sha256', 'diagnostics')}, ensure_ascii=False), flush=True)
    except BaseException as error:
        failed = {'status': 'FAIL_ENGINEERING_GATE', 'error': repr(error), 'traceback': traceback.format_exc(),
                  'checked_unix': time.time(), 'wall_seconds': time.time()-started, 'evidence_sha256': evidence}
        failed_path = args.output.with_name(args.output.stem+'.failed.'+str(int(time.time()))+'.json')
        save(failed_path, failed)
        print(json.dumps(failed), file=sys.stderr, flush=True)
        raise


if __name__ == '__main__':
    main()
