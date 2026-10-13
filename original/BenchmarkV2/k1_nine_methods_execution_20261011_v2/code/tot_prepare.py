"""Read-only legacy prefix audit; freeze new property-free ToT before GPU work."""
import argparse
import json
import re
import time
from functools import lru_cache
from pathlib import Path
from tot_protocol import *


@lru_cache(None)
def source_shards(model, task):
    base = BV2/('next3_ten_tasks_run01_v1' if model == 'mistral' else 'p6_llama_main_ten_tasks_v1')/'runs'
    pattern = (r'p6_tot20__'+task+r'__(\d+)_(\d+)' if model == 'mistral'
               else r'llama_p6_tot20_'+task+r'_(\d+)_(\d+)')
    result = []
    for directory in base.iterdir():
        match = re.fullmatch(pattern, directory.name)
        if match:
            result.append((int(match[1]), int(match[2]), directory/('run01' if model == 'mistral' else '')))
    assert result, (model, task)
    return result


@lru_cache(None)
def shard_metadata(model, path):
    directory = Path(path)
    records = read(directory/'inputs.json')
    identity = read(directory/'model_identity.json')
    spec = read(RUNTIMES[model])
    assert identity['load_spec'] == spec, (model, path, 'model_identity')
    status = identity['adapter_status']
    assert status['enabled'] is True and status['active_adapters'] == ['default'] and status['merged_adapters'] == []
    return {r['input_idx']: r for r in records}, {'path': str(directory/'model_identity.json'),
        'sha256': sha(directory/'model_identity.json'), 'load_spec': spec,
        'tokenizer': identity['tokenizer'], 'versions': identity['versions']}


def load_prefix(model, record, objectives):
    idx, task = record['input_idx'], record['task_id']
    dirs = [(start, directory) for start, end, directory in source_shards(model, task) if start <= idx < end]
    assert len(dirs) == 1, (model, task, idx, dirs)
    start, directory = dirs[0]
    source_records, identity = shard_metadata(model, str(directory))
    assert source_records[idx] == record, (model, task, idx, 'input_identity')
    prefix_path = directory/('candidates/input{}_depth1.json'.format(idx) if model == 'mistral'
                            else 'generated/{:04d}_initial.json'.format(idx))
    candidates = read(prefix_path)
    assert len(candidates) == 5 and [c['original_slot_id'] for c in candidates] == list(range(5))
    calls = []
    for branch, candidate in enumerate(candidates):
        assert candidate['input_uid'] == record['input_uid'] and candidate['task_id'] == task
        assert candidate['input_idx'] == idx and candidate['model_id'] == MODEL_IDS[model]
        assert candidate['stage'] == 'depth1' and candidate['node'] == 'b'+str(branch)
        assert candidate['source_smiles'] == record['source_smiles']
        assert candidate['parse_status'] == parse_smiles(candidate['raw_text'])
        assert candidate['raw_smiles'] == candidate['parse_status']['parsed_smiles']
        call_idx = (idx-start)*5+branch
        call_path = directory/'calls'/('{:03d}_result.json'.format(call_idx) if model == 'mistral'
                                       else '{:05d}.json.gz'.format(call_idx))
        call = read(call_path)
        assert call['completed'] and call['exception'] is None
        assert call['context']['input_uid'] == record['input_uid']
        assert call['context']['input_idx'] == idx and call['context']['stage'] == 'depth1'
        assert call['context']['branch'] == branch
        assert call['seed'] == 601+idx*100+branch
        assert call['prompt'] == initial_prompt(record, objectives, branch)
        assert call['prompt_digest'] == digest(call['prompt'])
        config = call['generation_config_passed'] if model == 'mistral' else call['effective_decoding']
        for key, value in {'do_sample': True, 'temperature': 1.0, 'top_p': 1.0, 'top_k': 0,
                           'num_beams': 1, 'num_return_sequences': 1, 'max_new_tokens': 160}.items():
            assert config[key] == value, (model, task, idx, key)
        assert len(call['outputs']) == 1
        output = call['outputs'][0]
        text = output['decoded_continuation_untrimmed'] if model == 'mistral' else output['raw_text']
        assert text == candidate['raw_text']
        eos, bos = ((2, 1) if model == 'mistral' else (128009, 128000))
        assert call['input_ids'][0] == bos and config['eos_token_id'] == eos
        generated = (output['continuation_ids'] if model == 'mistral' else output['generated_ids'])
        assert 1 <= len(generated) <= 160
        calls.append({'path': str(call_path), 'sha256': sha(call_path), 'seed': call['seed'],
                      'prompt': call['prompt'], 'prompt_digest': call['prompt_digest'],
                      'input_ids': call['input_ids'], 'output_tokens': (output['tokens_through_first_eos']
                          if model == 'mistral' else output['output_tokens']),
                      'cuda_wall_seconds': call['cuda_wall_seconds']})
    return {'record': record, 'initial_candidates': candidates,
            'initial_provenance': {'source_prefix_path': str(prefix_path), 'source_prefix_sha256': sha(prefix_path),
                                   'model_identity': identity, 'calls': calls,
                                   'assertions': ['same_model', 'same_manifest', 'same_prompt', 'same_seed',
                                                  'same_decoding', 'same_raw_response', 'same_parser', 'depth1_only']}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=tuple(MODEL_IDS), action='append')
    parser.add_argument('--task', choices=TASKS, action='append')
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    started = time.time()
    contract_path = ROOT/'config/tot_contract.json'
    if contract_path.exists():
        assert read(contract_path) == code_contract()
    else:
        save(contract_path, code_contract())
    receipts = []
    for model in args.model or MODEL_IDS:
        for task in args.task or TASKS:
            path = ROOT/'config'/f'tot_prefix_{model}_{task}.json'
            records = manifest(task)
            objectives = task_spec(task)['objectives']
            if path.exists():
                rows = read(path)
                assert [r['record'] for r in rows] == records
            else:
                rows = [load_prefix(model, record, objectives) for record in records]
                save(path, rows)
            receipt = {'model': model, 'task': task, 'inputs': len(rows), 'initial_candidates': 5*len(rows),
                       'path': str(path), 'sha256': sha(path)}
            receipts.append(receipt)
            print(json.dumps(receipt), flush=True)
    if args.freeze:
        assert not args.model and not args.task
        assert sum(r['inputs'] for r in receipts) == 7810
        files = ['config/tot_contract.json']+[str(Path(r['path']).relative_to(ROOT)) for r in receipts]
        files += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'code').glob('tot_*.py'))]
        files += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'tests').glob('test_tot_*.py'))]
        save(ROOT/'config/tot_freeze.json', {'status': 'FROZEN_BEFORE_REAL_GPU_PILOT',
             'created_unix': time.time(), 'files': {p: sha(ROOT/p) for p in files},
             'prefix_audit': receipts, 'full_inputs': 7810, 'reused_initial_candidates': 39050,
             'new_responses_planned': 195250, 'wall_seconds': time.time()-started})


if __name__ == '__main__':
    main()
