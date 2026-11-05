"""Resumable C20 continuation, synthetic development checks and full-input gate."""
import argparse
import fcntl
import os
import time
import traceback
from pathlib import Path
import protocol as p


def execute(model, record, backend, contract_sha, sources, stage):
    directory = p.ROOT / 'runs' / model / record['task_id'] / ('%04d' % record['input_idx'])
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        complete = directory / 'complete.json'
        if complete.exists():
            result = p.read(complete)
            assert result['contract_sha256'] == contract_sha and result['record'] == record
            assert result['model_id'] == p.MODEL_IDS[model] and len(result['candidates']) == 20
            return result
        evidence, families, old_dir = p.old_evidence(model, record, sources)
        prefix, g, pv = evidence['initial_prefix'], evidence['complete'], evidence['parent_vote']
        if backend.model is None:
            backend.start()
        backend.verify_prefix_tokens(prefix)
        initial = prefix['initial_candidates']
        parent_vote = p.ranking(pv['outputs'], 5)
        parent_indices = parent_vote['ranking'][:4]
        objectives = p.old.task_spec(record['task_id'])['objectives']
        children, calls, reused = [], [], []
        for rank, parent_idx in enumerate(parent_indices):
            parent = initial[parent_idx]
            prompt = p.old.child_prompt(record, objectives, parent)
            if parent['candidate_id'] in families:
                family = families[parent['candidate_id']]
                reused.extend(child['candidate_id'] for child in family)
            else:
                call = backend.generate([prompt] * 5, 'missing_children',
                    p.new_seed(model, record['task_id'], record['input_idx'], 'missing_children', parent_idx),
                    directory / ('children_parent_%d.json' % parent_idx))
                calls.append(call)
                family = [dict(raw_text=out['raw_text'],
                    raw_smiles=p.old.parse_smiles(out['raw_text'])['parsed_smiles'],
                    call_path=out['call_path'], batch_row=i, generation_seed=out['seed'],
                    prompt_sha256=p.digest(prompt)) for i, out in enumerate(call['outputs'])]
            assert len(family) == 5
            for branch, source in enumerate(family):
                frontier_index = rank * 5 + branch
                identity = dict(execution=p.EXECUTION, model_id=p.MODEL_IDS[model], method_id=p.METHOD,
                    task_id=record['task_id'], input_uid=record['input_uid'],
                    parent_candidate_id=parent['candidate_id'], branch=branch)
                candidate = dict(identity, candidate_id='tot-c20-' + p.digest(identity),
                    input_idx=record['input_idx'], source_smiles=record['source_smiles'],
                    raw_text=source['raw_text'], raw_smiles=source['raw_smiles'],
                    frontier_index=frontier_index, parent_initial_index=parent_idx, parent_rank=rank,
                    call_path=source['call_path'], batch_row=source['batch_row'],
                    generation_seed=source['generation_seed'], prompt_sha256=p.digest(prompt),
                    original_candidate_id=source.get('candidate_id'),
                    reused_generation='candidate_id' in source)
                assert candidate['prompt_sha256'] == source['prompt_sha256']
                children.append(candidate)
        assert len(children) == 20 and len({c['candidate_id'] for c in children}) == 20
        terminal_prompt = p.judge_prompt(record, objectives, children)
        terminal_call = backend.generate([terminal_prompt] * 5, 'terminal_vote',
            p.new_seed(model, record['task_id'], record['input_idx'], 'terminal_vote'),
            directory / 'terminal_vote.json')
        calls.append(terminal_call)
        terminal_vote = p.ranking(terminal_call['outputs'], 20)
        candidates = [dict(children[i], original_slot_id=slot) for slot, i in enumerate(terminal_vote['ranking'])]
        old_child_call = evidence['children']
        costs = dict(logical_model_responses_total=35, logical_initial_responses=5,
            logical_parent_votes=5, logical_children=20, logical_terminal_votes=5,
            reused_initial_responses=5, reused_parent_votes=5, reused_child_responses=len(reused),
            new_child_responses=20 - len(reused), new_terminal_votes=5,
            new_physical_generate_calls=len(calls),
            new_physical_cuda_wall_seconds=sum(c['physical_cuda_wall_seconds'] for c in calls),
            new_prompt_tokens_expanded=sum(o['input_tokens'] for c in calls for o in c['outputs']),
            new_output_tokens_through_eos=sum(o['output_tokens'] for c in calls for o in c['outputs']),
            reused_initial_gpu_seconds=g['costs']['reused_initial_gpu_seconds'],
            reused_parent_gpu_seconds=pv['physical_cuda_wall_seconds'],
            old_child_batch_gpu_seconds=old_child_call['physical_cuda_wall_seconds'],
            old_child_batch_note='Original physical15-row batch; some branches may be excluded after parser repair',
            property_queries_during_generation=0)
        result = dict(schema_version='BV2-TOT-C20-v1', status='GENERATION_COMPLETE_PROPERTIES_PENDING',
            contract_sha256=contract_sha, record=record, **{k: record[k] for k in (
                'task_id', 'input_uid', 'input_idx', 'source_smiles', 'source_properties')},
            model_id=p.MODEL_IDS[model], method_id=p.METHOD, candidates=candidates,
            selected_candidate=candidates[0], children=children, parent_vote=parent_vote,
            selected_parent_indices=parent_indices, terminal_vote=terminal_vote, costs=costs,
            generation_source_path=str(complete), source_stage=stage, completed_unix=time.time(),
            provenance=dict(old_complete_path=str(old_dir / 'complete.json'),
                old_complete_sha256=sources[str(old_dir / 'complete.json')],
                old_prefix_sha256=sources[str(old_dir / 'initial_prefix.json')],
                reused_parent_call_path=str(old_dir / 'parent_vote.json'),
                reused_parent_call_sha256=sources[str(old_dir / 'parent_vote.json')],
                reused_children_call_path=str(old_dir / 'children.json'),
                reused_children_call_sha256=sources[str(old_dir / 'children.json')],
                terminal_vote_call_path=str(directory / 'terminal_vote.json'),
                terminal_vote_sha256=p.sha(directory / 'terminal_vote.json'),
                candidate_property_access=False, external_validity_filter=False))
        p.save(complete, result)
        return result


def development(model, backend, contract_sha):
    # Fixed synthetic chemistry examples; no benchmark input or success labels.
    molecules = ['CCO', 'CCCO', 'CCCCO', 'CC(C)O', 'CC(=O)O', 'CC(=O)N', 'CCN',
        'CCCN', 'CCCCN', 'C1CCCCC1', 'c1ccccc1', 'Oc1ccccc1', 'Nc1ccccc1',
        'COc1ccccc1', 'CCc1ccccc1', 'CCOC', 'CCOCC', 'CCNCC', 'CC(C)N', 'CC(C)C']
    directory = p.ROOT / 'development' / model
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'complete.json').exists():
        previous = p.read(directory / 'complete.json')
        assert previous['contract_sha256'] == contract_sha
        return previous
    if backend.model is None:
        backend.start()
    results = []
    for index, n in enumerate((5, 20)):
        record = dict(source_smiles='CCO')
        objectives = [dict(property='qed', direction=1)]
        prompt = p.judge_prompt(record, objectives, [dict(raw_smiles=s) for s in molecules[:n]])
        call = backend.generate([prompt] * 5, 'development',
            p.new_seed(model, 'synthetic', index, 'development'), directory / ('judge_%d.json' % n))
        results.append(dict(candidates=n, votes=p.ranking(call['outputs'], n),
                            prompt_tokens=max(o['input_tokens'] for o in call['outputs']),
                            cuda_seconds=call['physical_cuda_wall_seconds']))
    receipt = dict(status='PASS_DEVELOPMENT_INTERFACE_EXECUTION', contract_sha256=contract_sha,
        examples='Fixed synthetic molecules; independent of benchmark; no expected best molecule',
        no_outcome_based_interface_selection=True, raw_output_parse_counts_descriptive_only=True,
        checks=results)
    p.save(directory / 'complete.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=tuple(p.MODEL_IDS))
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--unit-index', type=int)
    args = parser.parse_args()
    spec, contract_sha = p.validate_freeze()
    sources = p.read(spec['source_audit'])
    attempt = p.ROOT / 'attempts' / args.model / (os.environ.get('SLURM_JOB_ID', 'local') + '_' + str(time.time_ns()))
    attempt.mkdir(parents=True, exist_ok=True)
    from backend import Backend
    backend = Backend(args.model, spec['model_assembly'], contract_sha, attempt)
    started, completed = time.time(), []
    try:
        if args.probe:
            development(args.model, backend, contract_sha)
            units = [dict(task=t, start=0, end=4) for t in spec['probe_tasks']]
        else:
            gate = p.read(p.ROOT / 'reports' / (args.model + '_engineering_gate.json'))
            assert gate['status'] == 'PASS_PRODUCTION_ENGINEERING' and gate['contract_sha256'] == contract_sha
            assert args.unit_index is not None
            units = [p.read(p.ROOT / 'units.json')[args.unit_index]]
        for unit in units:
            for record in p.old.manifest(unit['task'])[unit['start']:unit['end']]:
                row = execute(args.model, record, backend, contract_sha, sources, 'probe' if args.probe else 'full')
                completed.append(dict(path=row['generation_source_path'], sha256=p.sha(row['generation_source_path']),
                    input_uid=row['input_uid'], task_id=row['task_id'], costs=row['costs'],
                    parent_valid_votes=row['parent_vote']['valid_votes'],
                    terminal_valid_votes=row['terminal_vote']['valid_votes']))
                print(p.json.dumps(dict(event='tot_c20_input_complete', model=args.model,
                    task=record['task_id'], input_idx=record['input_idx'], completed=len(completed))), flush=True)
        receipt = dict(status='PASS_PRODUCTION_ENGINEERING' if args.probe else 'COMPLETE_UNIT',
            contract_sha256=contract_sha, model=args.model, inputs=completed, input_count=len(completed),
            wall_seconds=time.time() - started, property_feedback=False,
            checks=['exact_old_source_hashes', 'parent_identity_and_prompt_cache', '20_terminal_slots',
                    'new_final_votes', 'K1_is_rank0', 'no_context_truncation', 'explicit_160_child_128_judge_caps'],
            no_success_or_parse_rate_threshold=True)
        p.save(attempt / 'complete.json', receipt)
        if args.probe:
            assert len(completed) == 12
            p.save(p.ROOT / 'reports' / (args.model + '_engineering_gate.json'), receipt)
    except BaseException as error:
        p.save(attempt / 'failure.json', dict(error=repr(error), traceback=traceback.format_exc(),
             completed=completed, wall_seconds=time.time() - started))
        raise


if __name__ == '__main__':
    main()
