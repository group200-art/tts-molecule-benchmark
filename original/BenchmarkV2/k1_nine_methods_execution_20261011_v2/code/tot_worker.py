"""Full-input LM-vote ToT generation; external property evaluation is separate."""
import argparse
import fcntl
import os
import time
import traceback
from pathlib import Path
from tot_protocol import *


def execute_input(model, prefix, objectives, backend, contract_sha, stage):
    record = prefix['record']
    idx, task, uid = record['input_idx'], record['task_id'], record['input_uid']
    directory = ROOT/'runs'/f'tot_{model}_{task}'/f'{idx:04d}'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        complete_path = directory/'complete.json'
        if complete_path.exists():
            completed = read(complete_path)
            assert completed['contract_sha256'] == contract_sha
            assert completed['input_uid'] == uid and completed['model_id'] == MODEL_IDS[model]
            assert completed['prefix_digest'] == digest(prefix)
            assert completed['status'] == 'complete_generation_pending_property_evaluation'
            return completed
        if backend.model is None:
            backend.start()
        backend.verify_prefix_tokens(prefix)
        initial = prefix['initial_candidates']
        prefix_path = directory/'initial_prefix.json'
        if prefix_path.exists():
            assert read(prefix_path) == prefix
        else:
            save(prefix_path, prefix)
        parent_prompt = judge_prompt(record, objectives, initial)
        parent_call = backend.generate([parent_prompt]*5, 'parent_vote',
            batch_seed(model, task, idx, 'parent_vote'), directory/'parent_vote.json')
        parent_vote = vote_ranking(parent_call['outputs'], 5)
        parent_indices = parent_vote['ranking'][:3]
        parents = [initial[i] for i in parent_indices]
        prompts = [child_prompt(record, objectives, parent) for parent in parents for _ in range(5)]
        child_call = backend.generate(prompts, 'children',
            batch_seed(model, task, idx, 'children'), directory/'children.json')
        children = []
        for i, output in enumerate(child_call['outputs']):
            parent = parents[i//5]
            parsed = parse_smiles(output['raw_text'])
            identity = {'model_id': MODEL_IDS[model], 'method_id': 'tot_lm_vote', 'task_id': task,
                        'input_uid': uid, 'stage': 'depth2', 'node': f'p{i//5+1}:b{i%5}',
                        'original_slot_id': 5+i, 'contract_sha256': contract_sha}
            candidate = dict(identity, candidate_id='candidate-v2-'+digest(identity),
                input_idx=idx, raw_text=output['raw_text'], raw_smiles=parsed['parsed_smiles'],
                parse_status=parsed, source_smiles=record['source_smiles'],
                parent_candidate_id=parent['candidate_id'], parent_original_slot_id=parent['original_slot_id'],
                parent_rank=i//5+1, branch=i%5, call_path=output['call_path'], batch_row=i,
                prompt_sha256=digest(prompts[i]), generation_seed=output['seed'])
            children.append(candidate)
        assert len(children) == 15 and [c['original_slot_id'] for c in children] == list(range(5, 20))
        terminal_prompt = judge_prompt(record, objectives, children)
        terminal_call = backend.generate([terminal_prompt]*5, 'terminal_vote',
            batch_seed(model, task, idx, 'terminal_vote'), directory/'terminal_vote.json')
        terminal_vote = vote_ranking(terminal_call['outputs'], 15)
        chosen_index = terminal_vote['ranking'][0]
        chosen = children[chosen_index]
        selected = dict(chosen, source_original_slot_id=chosen['original_slot_id'], original_slot_id=0)
        calls = [parent_call, child_call, terminal_call]
        result = {'schema_version': 'BV2-TOT-GENERATION-ENDPOINT-v2',
            'status': 'complete_generation_pending_property_evaluation', 'contract_sha256': contract_sha,
            'model_id': MODEL_IDS[model], 'method_id': 'tot_lm_vote', 'task_id': task,
            'input_uid': uid, 'input_idx': idx, 'source_smiles': record['source_smiles'],
            'source_properties': record['source_properties'], 'selected_candidate': selected,
            'prefix_digest': digest(prefix), 'origin_stage': stage, 'completed_unix': time.time(),
            'initial_prefix_path': str(prefix_path), 'initial_candidates': initial, 'children': children,
            'parent_vote': parent_vote, 'selected_parent_indices': parent_indices, 'terminal_vote': terminal_vote,
            'selection_provenance': {'rule': 'LM-vote(5) at both frontiers; keep3 initial, keep1 last15; ties original order',
                'source_path': str(complete_path), 'contract_sha256': contract_sha,
                'selected_terminal_frontier_index': chosen_index, 'selected_candidate_id': chosen['candidate_id'],
                'parent_vote_call_path': str(directory/'parent_vote.json'),
                'terminal_vote_call_path': str(directory/'terminal_vote.json'),
                'candidate_property_access': False, 'external_validity_filter': False},
            'costs': {'reused_initial_responses': 5, 'new_molecule_responses': 15, 'judge_responses': 10,
                'new_physical_generate_calls': 3, 'logical_model_responses_total': 30,
                'reused_initial_gpu_seconds': sum(c['cuda_wall_seconds'] for c in prefix['initial_provenance']['calls']),
                'reused_initial_prompt_tokens': sum(len(c['input_ids']) for c in prefix['initial_provenance']['calls']),
                'reused_initial_output_tokens': sum(c['output_tokens'] for c in prefix['initial_provenance']['calls']),
                'new_physical_cuda_wall_seconds': sum(c['physical_cuda_wall_seconds'] for c in calls),
                'new_prompt_tokens_expanded': sum(o['input_tokens'] for c in calls for o in c['outputs']),
                'new_output_tokens_through_eos': sum(o['output_tokens'] for c in calls for o in c['outputs']),
                'property_queries_during_generation': 0}}
        save(complete_path, result)
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=tuple(MODEL_IDS))
    parser.add_argument('--task', required=True, choices=TASKS)
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--end', type=int, required=True)
    parser.add_argument('--stage', required=True, choices=('pilot', 'full'))
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    started = time.time()
    contract, contract_sha = validate_freeze()
    records = manifest(args.task)
    assert 0 <= args.start < args.end <= len(records)
    prefixes = read(ROOT/'config'/f'tot_prefix_{args.model}_{args.task}.json')
    assert [p['record'] for p in prefixes] == records
    objectives = task_spec(args.task)['objectives']
    selected_prefixes = prefixes[args.start:args.end]
    if args.validate_only:
        print(json.dumps({'status': 'PASS_FROZEN_INPUTS', 'model': args.model, 'task': args.task,
                          'inputs': len(selected_prefixes), 'contract_sha256': contract_sha}))
        return
    job_id = os.environ.get('SLURM_JOB_ID', 'no_allocation')
    job_directory = ROOT/'runs'/'tot_jobs'/f'{args.model}_{job_id}_{args.task}_{args.start}_{args.end}'
    # Each scheduler invocation uses one model load; input-level evidence is shared by pilot and full.
    job_directory.mkdir(parents=True, exist_ok=True)
    from tot_backend import Backend
    backend = Backend(args.model, contract, contract_sha, job_directory)
    completed = []
    try:
        for prefix in selected_prefixes:
            result = execute_input(args.model, prefix, objectives, backend, contract_sha, args.stage)
            completed.append({'input_uid': result['input_uid'], 'input_idx': result['input_idx'],
                              'path': result['selection_provenance']['source_path'], 'costs': result['costs'],
                              'valid_parent_votes': result['parent_vote']['valid_votes'],
                              'valid_terminal_votes': result['terminal_vote']['valid_votes']})
            print(json.dumps({'event': 'tot_input_complete', 'model': args.model, 'task': args.task,
                              'input_idx': result['input_idx'], 'completed': len(completed),
                              'assigned': len(selected_prefixes)}), flush=True)
        save(job_directory/'completion.json', {'status': 'complete', 'model': args.model, 'task': args.task,
            'start': args.start, 'end': args.end, 'stage': args.stage, 'contract_sha256': contract_sha,
            'inputs': completed, 'wall_seconds': time.time()-started,
            'model_load_seconds': getattr(backend, 'load_seconds', 0.0)})
    except BaseException as error:
        failure_path = job_directory/'failure.json'
        if not failure_path.exists():
            save(failure_path, {'status': 'failed', 'error': repr(error), 'traceback': traceback.format_exc(),
                'completed': completed, 'contract_sha256': contract_sha, 'wall_seconds': time.time()-started})
        raise


if __name__ == '__main__':
    main()
