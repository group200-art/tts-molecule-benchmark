"""CPU-only freeze and complete 3905-input audit of the reusable Llama r1 prefix."""
import ast
import time
from common import *


def audit_reuse():
    templates = read(LLAMA / 'configs/p5_prompts.json')
    assert templates == read(REPO / 'BenchmarkV2/method_review_P5_v1/execution_v1/configs/prompts.json')
    runtime = read(LLAMA / 'configs/runtime.json')
    old_method = read(LLAMA / 'configs/full.json')['methods']['self_refine']
    sources, units, counts, property_rows = {}, {}, {}, 0
    for task in TASKS:
        prep_path = K1 / 'runs/self_refine_llama' / task / 'prepared.json.gz'
        result_path = K1 / 'results' / ('self_refine_llama__' + task + '.json.gz')
        prepared, results = read(prep_path), read(result_path)
        records = manifest(task)
        assert prepared['records'] == records and len(results) == len(records)
        sources[str(prep_path)], sources[str(result_path)] = sha(prep_path), sha(result_path)
        reused = []
        for record, candidate, provenance, endpoint in zip(records, prepared['candidates'], prepared['provenance'], results):
            path = Path(provenance['trajectory_path'])
            assert sha(path) == provenance['trajectory_sha256']
            tr = read(path)
            assert tr['record'] == record and tr['model_id'] == 'gellmo_p6_llama'
            assert tr['rep_seed'] == 51001 and tr['complete'] and tr['inference_tool_calls'] == 0
            assert tr['task_text'] == task_text(record)
            assert tr['answers'][1]['raw_text'] == candidate['raw_text']
            assert legacy_candidate(record, 51001, 1, tr['answers'][1], tr['scientific_identity_sha256']) == candidate
            assert candidate['parse'] == parse_answer(candidate['raw_text'])
            source = Path(provenance['source_path'])
            assert sha(source) == provenance['source_sha256'] and read(source) == tr['answers'][1]
            answers, feedbacks = [], []
            for position, (role, r) in enumerate(ROLES):
                c = tr['calls'][position]
                assert c == (tr['feedbacks'][r] if role == 'feedback' else tr['answers'][r])
                assert (c['role'], c['r']) == (role, r)
                assert c['seed'] == call_seed(51001, task, record['input_uid'], role, r)
                assert c['max_new_tokens'] == old_method['role_caps'][role]
                prompt = prompt_for(role, tr['task_text'], answers, feedbacks, r, templates)
                assert c['prompt_digest'] == digest(prompt)
                (feedbacks if role == 'feedback' else answers).append(c)
            assert provenance['prefix_generation_cost']['model_calls'] == 4
            selected = endpoint['selected_candidate']
            assert endpoint['input_uid'] == record['input_uid'] and endpoint['model_id'] == 'gellmo_p6_llama'
            assert selected['candidate_id'] == candidate['candidate_id'] and selected['raw_text'] == candidate['raw_text']
            assert selected['raw_smiles'] == candidate['raw_smiles'] and selected['rep_seed'] == 51001
            assert selected['source_original_slot_id'] == 1 and selected['original_slot_id'] == 0
            assert selected['property_error'] is None or selected['known_molecular_property_failure']
            assert not (selected['rdkit_valid'] and selected['properties'] is None and not selected['known_molecular_property_failure'])
            unit = path.parents[2]
            if str(unit) not in units:
                identity_path = unit / 'model_identity.json'
                identity = read(identity_path)
                assert identity['model_id'] == 'gellmo_p6_llama' and identity['load_spec'] == runtime
                assert identity['tokenizer']['bos'] == 128000 and identity['tokenizer']['eos'] == 128009
                assert identity['tokenizer']['is_fast'] and not identity['tokenizer']['chat_template_applied']
                assert identity['adapter_status']['active_adapters'] == ['default']
                assert identity['adapter_status']['merged_adapters'] == []
                # One complete four-call prefix per historical physical unit checks actual decoding.
                for c in tr['calls'][:4]:
                    raw = read(c['call_path'])
                    assert raw['completed'] and raw['prompt_digest'] == c['prompt_digest']
                    assert all(raw['effective_decoding'][k] == v for k, v in old_method['decoding'].items())
                    assert raw['effective_decoding']['max_new_tokens'] == old_method['role_caps'][c['role']]
                    assert (raw['effective_decoding']['bos_token_id'], raw['effective_decoding']['eos_token_id'],
                            raw['effective_decoding']['pad_token_id']) == (128000, 128009, 128009)
                    sources[c['call_path']] = sha(c['call_path'])
                sources[str(identity_path)] = sha(identity_path)
                units[str(unit)] = dict(first_uid=record['input_uid'], actual_decode_calls_checked=4)
            sources[str(path)], sources[str(source)] = provenance['trajectory_sha256'], provenance['source_sha256']
            reused.append(dict(record=record, repeat=0, rep_seed=51001,
                               candidate=selected, provenance=provenance))
            property_rows += 1
        assert len(reused) == len(records)
        counts[task] = len(reused)
        save(ROOT / 'reuse' / (task + '.json.gz'), reused)
        print(json.dumps(dict(event='reuse_audited', task=task, inputs=len(reused))), flush=True)
    assert sum(counts.values()) == property_rows == 3905
    report = dict(status='PASS_FULL3905_SEED51001_R1_PREFIX_REUSE', counts=counts,
                  reusable_endpoints=3905, prefix_calls_audited=3905*4,
                  historical_units_checked=len(units), actual_decoding_call_spots=len(units)*4,
                  same_prompt_parser_controller=True, no_quality_selection=True,
                  note='Existing serial seed51001 retained; new batched random streams are not bitwise serial continuations.',
                  new_endpoints_required=3905*19, new_response_sequences=3905*19*4)
    save(ROOT / 'reports/reuse_sources.json', sources)
    save(ROOT / 'reports/reuse_audit.json', report)
    return old_method


def main():
    if (ROOT / 'reports/freeze.json').exists():
        verify()
        print('Already prepared; frozen identities verified.')
        return
    method = audit_reuse()
    units = []
    for task in TASKS:
        rows = manifest(task)
        for start in range(0, len(rows), 80):
            part = rows[start:start+80]
            name = 'llama_sr_r1_{}_{}_{:04d}'.format(task, str(start).zfill(4), start+len(part))
            batches = [dict(batch_id=name+'_b'+str(i//4).zfill(2), records=part[i:i+4])
                       for i in range(0, len(part), 4)]
            units.append(dict(index=len(units), unit_id=name, task_id=task, n_inputs=len(part), batches=batches))
    assert sum(u['n_inputs'] for u in units) == 3905
    keys, seeds = [], []
    for unit in units:
        for batch in unit['batches']:
            keys.extend((r['task_id'], r['input_uid'], s) for r, s in ordered_lanes(batch))
            seeds.extend(batch_seed(batch, *role) for role in ROLES)
    assert len(keys) == len(set(keys)) == 74195 and len(seeds) == len(set(seeds))
    save(ROOT / 'units.json', units)
    icl = read(ICL / 'contract.json')
    assert icl['model_assembly']['models']['llama'] == read(LLAMA / 'configs/runtime.json')
    science = read(MISTRAL / 'plan.json')['science']
    assert science['decoding'] == method['decoding'] and science['max_new_tokens'] == method['role_caps']
    contract = dict(execution_id=EXECUTION, model_id='gellmo_p6_llama', method='Self-Refine r1',
        status='FROZEN_FOR_ENGINEERING_PROBE_NOT_SUBMITTED', authorization='Root delegated preparation only; no GPU submission in this scope.',
        inputs=3905, output_slots=20, reuse_seed=51001, new_rep_seeds=NEW_SEEDS,
        output_order=list(range(51001, 51021)), roles=ROLES, refinements=1,
        decoding=method['decoding'], role_caps=method['role_caps'], context_limit=method['context_limit'],
        prompt_limit=method['prompt_limit'], terminal='answers[1]; include feedback1 before fixed stop',
        parser='Original Llama SR first complete case-sensitive SMILES tag; no repair, no validity fallback',
        inference_properties=False, state='Each input/rep_seed owns separate answers and feedbacks lists.',
        model_assembly=icl['model_assembly'], physical_batch='4 consecutive task-manifest inputs x19 new replicate IDs =76 lanes; final batch smaller',
        rng='SHA256(execution,batch_id,ordered(task,input_uid,rep_seed),role,r), first15hex modulo2147483647; reset every physical call',
        batch_equivalence='Sampling policy retained; no bitwise equivalence to serial streams or Mistral batches claimed.',
        batch_changes='No automatic OOM downsize, quality retry, row sorting, or score-based tuning; failed infrastructure blocks completion.',
        probe_unit_indices=[u['index'] for u in units if u['task_id'] in ('BDP','BDPQ','BHMQ') and u['batches'][0]['records'][0]['input_idx']==0],
        probe='First frozen batch in each probe unit:12 inputs x19 new trajectories=228 terminals/912 responses; reused unchanged in full run.',
        expected_new_terminals=74195, expected_new_response_sequences=296780,
        max_gpus_this_scope=1, total_project_gpu_limit=2,
        property_evaluation='Separate CPU original GeLLMO producers/precision; generation completion is not benchmark completion.',
        k1='Existing fixed seed51001 endpoint; no cross-seed reranking or best-seed selection.',
        templates_path=str(LLAMA/'configs/p5_prompts.json'), units_sha256=sha(ROOT/'units.json'),
        reuse_audit_sha256=sha(ROOT/'reports/reuse_audit.json'))
    save(ROOT / 'contract.json', contract)
    paths = set((ROOT/'code').glob('*.py')) | set((ROOT/'code').glob('*.sbatch'))
    paths |= {ROOT/'contract.json', ROOT/'units.json', ROOT/'reports/reuse_audit.json', ROOT/'reports/reuse_sources.json'}
    paths |= set((ROOT/'reuse').glob('*.json.gz'))
    paths |= {LLAMA/p for p in ('src/contracts.py','src/self_refine.py','src/llama_generation.py','configs/full.json',
              'configs/runtime.json','configs/p5_prompts.json','configs/mistral_generation_defaults.json',
              'evidence/read_set.json','evidence/model_stat_inventory.json')}
    paths |= {K1/'code/tot_protocol.py', K1/'code/tot_backend.py', ICL/'contract.json', MISTRAL/'plan.json'}
    paths |= {Path(task_contract(t)['manifest_path']) for t in TASKS}
    paths |= {REPO/p for p in ('prompter.py','config.py','templates/simple.json','BenchmarkV2/registries/task_registry.draft.json')}
    for folder in (Path(contract['model_assembly']['models']['llama']['base_path']),
                   Path(contract['model_assembly']['models']['llama']['adapter_path'])):
        paths |= {p for p in folder.iterdir() if p.is_file() and p.suffix in ('.json','.model')}
    save(ROOT/'reports/freeze.json', dict(status='FROZEN_CPU_PREPARATION', sha256={str(p):sha(p) for p in sorted(paths)}))
    verify()
    print(json.dumps(dict(status='PREPARED_NOT_SUBMITTED', new_terminals=74195, units=len(units))))


if __name__ == '__main__':
    main()
