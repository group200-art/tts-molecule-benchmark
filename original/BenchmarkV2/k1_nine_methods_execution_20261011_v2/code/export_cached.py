"""Export seven accepted K=1 methods from raw cached outputs, then reevaluate.

Selection never reads legacy success-selected representative fields. All scientific
inputs are read-only. File hashes, exact UID coverage, output identity and native
metric/coverage equivalences are audited before each completed group is published.
"""
import argparse
import copy
import gzip
import hashlib
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
from cache_metrics import B, ROOT, REPO, KERNEL, TASK_REGISTRY, evaluate_registered, aggregate_registered

TASKS = ['BDP', 'BDQ', 'BPQ', 'DPQ', 'BDPQ', 'MPQ', 'BDMQ', 'BHMQ', 'BMPQ', 'HMPQ']
METHODS = ['beam20', 'sampling_verifier20', 'got', 's1', 'ev', 'fewshot', 'reflexion']
PAIRED = B / 'p6_llama_failure_recovery_20261003_v1/results'
M_GOT = B / 'method_review_P8_v1/design_after_p9_v1/execution_v1'
L_OLD = B / 'p6_llama_main_ten_tasks_v1'
L_REC = PAIRED.parent
SOURCES = {}
EXCEPTIONS = []


def track(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    item = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    if str(path) in SOURCES and SOURCES[str(path)] != item:
        raise ValueError('Source changed during export: ' + str(path))
    SOURCES[str(path)] = item
    return data


def read(path):
    return json.loads(track(path))


def lines(path):
    path = Path(path)
    data = track(path)
    if path.suffix == '.gz':
        data = gzip.decompress(data)
    return [json.loads(x) for x in data.splitlines() if x.strip()]


def write_json(path, value):
    path = Path(path)
    assert ROOT in path.resolve().parents
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    os.replace(temp, path)


def known_failure(candidate):
    """Only the already accepted molecule-level failures; not any cache miss."""
    representations = candidate.get('canonical_representations') or []
    inf_ok = any(x.get('environment') == 'gellmo_inf' and x.get('valid') is True for x in representations)
    eval_bad = any(x.get('environment') == 'gellmo_eval_py36' and x.get('valid') is False for x in representations)
    if inf_ok and eval_bad:
        return 'accepted_rdkit_2024_valid_2020_unsupported'
    if candidate.get('environment_validity_disagreement') is True:
        return 'accepted_explicit_environment_validity_disagreement'
    error = candidate.get('property_error')
    if (isinstance(error, dict) and error.get('exception_type') == 'KekulizeException'
            and error.get('category') == 'rdkit_kekulization_after_initial_parse'
            and error.get('initial_parse_valid') is True):
        return 'accepted_explicit_downstream_kekulize_exception'
    return None


def verify_candidate(candidate, source, context):
    if type(candidate['rdkit_valid']) is not bool or not isinstance(candidate['raw_smiles'], str):
        raise ValueError('Malformed candidate: ' + str(context))
    chemical = candidate['rdkit_valid'] and bool(candidate['raw_smiles'].strip())
    finite = KERNEL['finite_values'](candidate.get('properties'), list(source))
    if chemical and not finite:
        reason = known_failure(candidate)
        if reason is None:
            raise ValueError('Unresolved property absence: ' + str(context))
        EXCEPTIONS.append(dict(context, candidate_id=candidate['candidate_id'], reason=reason))
    return bool(chemical and finite)


def sampling_select(candidates, source, context=None):
    if len(candidates) != 20 or [c['original_slot_id'] for c in candidates] != list(range(20)):
        raise ValueError('Sampling requires the unchanged original slots 0..19')
    checked = []
    for c in candidates:
        usable = verify_candidate(c, source, context or {})
        accepted = usable and KERNEL['strictly_improves'](source, c['properties'], list(source))
        checked.append({'original_slot_id': c['original_slot_id'], 'candidate_id': c['candidate_id'],
                        'accepted': bool(accepted), 'property_eligible': usable,
                        'known_molecular_failure': known_failure(c) if not usable else None})
        if accepted:
            return c, {'rule': 'first_task_verifier_pass_in_original_slots_0_19',
                       'logical_validation_slots': len(checked), 'validation_trace': checked,
                       'fallback': False, 'actual_generation_budget_per_input': 40,
                       'declared_verification_pool': 20, 'success_oracle_equivalence_disclosed': True}
    return candidates[0], {'rule': 'first_task_verifier_pass_in_original_slots_0_19',
                           'logical_validation_slots': 20, 'validation_trace': checked,
                           'fallback': True, 'actual_generation_budget_per_input': 40,
                           'declared_verification_pool': 20, 'success_oracle_equivalence_disclosed': True}


def make_endpoint(model, method, record, candidate, provenance):
    context = dict(model_id=model, method_id=method, task_id=record['task_id'], input_uid=record['input_uid'])
    for key in ('input_uid', 'input_idx', 'source_smiles'):
        if key in candidate and key in record and candidate[key] != record[key]:
            raise ValueError('Candidate/input identity drift in ' + key)
    if 'model_id' in candidate and candidate['model_id'] != model:
        raise ValueError('Candidate/model identity drift')
    verify_candidate(candidate, record['source_properties'], context)
    c = copy.deepcopy(candidate)
    c['source_original_slot_id'] = c['original_slot_id']
    c['original_slot_id'] = 0
    # Generated proposal IDs and any original beam/graph/trial positions stay in c.
    evaluation = evaluate_registered(record['task_id'], record['input_uid'], record['source_properties'], [c], 1)
    assert evaluation['n_slots'] == 1
    return dict(schema_version='BV2-K1-ENDPOINT-v2', model_id=model, method_id=method,
                task_id=record['task_id'], input_uid=record['input_uid'], input_idx=record['input_idx'],
                source_properties=record['source_properties'], selected_candidate=c,
                selection_provenance=provenance, evaluation=evaluation)


def equality_check(current, previous):
    for key in ['SR_dir_success', 'SR_delta_success', 'RI_legacy', 'RI_abs_eps01',
                'input_valid', 'n_slots', 'n_valid_slots', 'n_eligible_slots']:
        if key in previous and current[key] != previous[key]:
            raise ValueError('Native metric drift in ' + key)


def indexed(rows, expected):
    out = {r['input_uid']: r for r in rows}
    assert len(out) == len(rows) and set(out) == {r['input_uid'] for r in expected}
    return out


def graph_candidate(graph):
    assert graph['event'] == 'complete' and graph['G'] == 12 and len(graph['nodes']) == 28
    ranked = graph['scores']['full']['ranked_node_ids']
    assert len(ranked) == 28 and len(set(ranked)) == 28 and graph['top20'] == ranked[:20]
    nid = ranked[0]
    node = next(n for n in graph['nodes'] if n['node_id'] == nid)
    p = graph['properties'][nid]
    c = dict(candidate_id=nid, original_slot_id=0, graph_rank=0, created_order=node['created_order'],
             raw_smiles=p['request_smiles'], rdkit_valid=p['rdkit_valid'], properties=p['properties'],
             property_row_exists=p['property_row_exists'], raw_properties=p.get('raw_properties'),
             property_error=p.get('property_error'),
             environment_validity_disagreement=p.get('environment_validity_disagreement', False))
    if p.get('rdkit_valid') is True and p.get('eval_valid') is False:
        c['environment_validity_disagreement'] = True
    return c


def mistral_graph_index():
    result = {}
    for base, kind in [(M_GOT, 'original_complete_reused'), (M_GOT / 'recovery_execution_v1', 'recovery_newly_completed')]:
        for path in sorted((base / 'workers').glob('*/unit_status.json')):
            status = read(path)
            for entry in status.get('completed_graphs', []):
                if entry['G'] != 12:
                    continue
                key = kind, status['unit']['task_id'], entry['input_uid']
                assert key not in result
                result[key] = base / entry['graph']
    accepted_path = M_GOT / 'recovery_execution_v1/results/per_input_topK.jsonl.gz'
    accepted = {(r['task_id'], r['input_uid']): r for r in lines(accepted_path) if r['G'] == 12 and r['K'] == 1}
    assert len(accepted) == 3905
    return result, accepted


def llama_graph_index():
    result = {}
    for base in [L_OLD, L_REC]:
        for directory in sorted((base / 'runs').glob('llama_got_*')):
            # Only completed units used in the accepted recovery report.
            if not (directory / 'metrics.json').is_file():
                continue
            task = directory.name.split('_')[2]
            for path in (directory / 'graphs').glob('*.json'):
                key = task, int(path.stem)
                if key in result:
                    raise ValueError('Ambiguous completed GoT graph: ' + str(key))
                result[key] = path
    assert len(result) == 3905, len(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', choices=['mistral', 'llama'], default=['mistral', 'llama'])
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=METHODS)
    args = parser.parse_args()
    started = time.monotonic()
    manifests = {task: lines(B / 'manifests' / (task + '.test.seen.jsonl')) for task in TASKS}
    assert sum(map(len, manifests.values())) == 3905
    for task, records in manifests.items():
        assert [r['input_idx'] for r in records] == list(range(len(records)))
        assert all(r['task_id'] == task for r in records)
    sources_for_metric = [B / 'src/metrics_cpu.py', B / 'registries/task_registry.draft.json',
                          B / 'next3_ten_tasks_run01_v1/src/metrics_entry.py', Path(__file__),
                          Path(__file__).with_name('cache_metrics.py'),
                          L_REC / 'reports/REPAIR_CONTRACT.md']
    for p in sources_for_metric:
        track(p)
    native_summary = {
        'mistral': {(r['method_id'], r['task_id']): r['metrics'] for r in read(PAIRED / 'mistral_main_metrics.json')},
        'llama': {(r['method_id'], r['task_id']): r['full_task_metrics'] for r in read(PAIRED / 'llama_main_metrics.json')},
    }
    group_summaries = []
    for short_model in args.models:
        model = 'gellmo_p6_' + short_model
        if short_model == 'mistral':
            s1_path = B / 'method_review_P9_v1/execution_v1/results/per_input_terminal_v1.jsonl'
            s1_rows = [r for r in lines(s1_path) if r['experiment_id'] == 'P9-B512-K1-v1'] if 's1' in args.methods else []
            got_index, got_accepted = mistral_graph_index() if 'got' in args.methods else ({}, {})
        else:
            got_index = llama_graph_index() if 'got' in args.methods else {}
        for method in args.methods:
            output_path = ROOT / 'results' / ('cached_' + short_model + '__' + method + '.jsonl.gz')
            temp_path = output_path.with_name(output_path.name + '.tmp.' + str(os.getpid()))
            output_count = 0
            with gzip.open(temp_path, 'wt', encoding='utf-8', compresslevel=5) as output:
                for task in TASKS:
                    expected = manifests[task]
                    old_method = 'sampling40_prefix20' if method == 'sampling_verifier20' else method
                    paths = []
                    rows = []
                    if short_model == 'llama':
                        paths = [PAIRED / 'by_task' / (old_method + '__' + task + '.json')]
                        rows = read(paths[0])['inputs']
                    elif method in ('beam20', 'sampling_verifier20'):
                        paths = [B / 'next3_ten_tasks_run01_v1/results/tasks' / (old_method + '__' + task + '.json')]
                        rows = read(paths[0])['inputs']
                    elif method == 's1':
                        paths = [s1_path]
                        rows = [r for r in s1_rows if r['task_id'] == task]
                    elif method == 'fewshot':
                        paths = [B / 'method_review_Pplus_v1/execution_v3/repair_v1/evaluation/final_eval/by_task' / ('s4_' + task + '.json')]
                        rows = read(paths[0])['inputs']
                    elif method == 'ev':
                        evroot = B / 'method_review_P10_v1/execution_vote_ev_v1/cpu'
                        paths = [evroot / phase / task / 'metrics.json' for phase in ('pilot', 'full')]
                        paths = [p for p in paths if p.exists()]
                        rows = [r for p in paths for r in read(p)['by_budget']['400']['inputs']]
                    elif method == 'reflexion':
                        paths = [REPO / 'Ppro/reflexion_remaining8_full_v1/results' / ('prefix_' + task + '_b3.json')]
                        rows = read(paths[0])['inputs']
                    elif method == 'got':
                        rows = [got_accepted[task, r['input_uid']] for r in expected]
                    by_uid = indexed(rows, expected)
                    evaluations = []
                    for record in expected:
                        old = by_uid[record['input_uid']]
                        assert old['source_properties'] == record['source_properties']
                        provenance = {'source_paths': [str(p.resolve()) for p in paths], 'no_new_generation': True,
                                      'properties_reused_at_original_comparison_precision': True}
                        context = dict(model_id=model, method_id=method, task_id=task, input_uid=record['input_uid'])
                        if method == 'got':
                            if short_model == 'mistral':
                                kind = old['record_provenance']['kind']
                                path = got_index[kind, task, record['input_uid']]
                            else:
                                path = got_index[task, record['input_idx']]
                            graph = read(path)
                            assert graph['record']['input_uid'] == record['input_uid']
                            assert graph['record']['source_properties'] == record['source_properties']
                            c = graph_candidate(graph)
                            if short_model == 'llama':
                                archived = old['candidates'][0]
                                for key in ('candidate_id', 'raw_smiles', 'rdkit_valid', 'properties'):
                                    assert archived[key] == c[key]
                            provenance.update(rule='native_full_graph_rank0', source_paths=[str(path)],
                                              ranked_node_id=graph['scores']['full']['ranked_node_ids'][0], graph_nodes=28)
                        elif method == 'sampling_verifier20':
                            c, details = sampling_select(old['candidates'], record['source_properties'], context)
                            provenance.update(details)
                        elif short_model == 'mistral' and method in ('s1', 'reflexion'):
                            c = dict(candidate_id=old['candidate_id'], original_slot_id=0,
                                     raw_smiles=old['raw_smiles'], rdkit_valid=old['rdkit_valid'],
                                     raw_properties=old.get('raw_properties'), properties=old['comparison_properties'],
                                     property_row_exists=old.get('property_row_exists', old['comparison_properties'] is not None))
                            provenance.update(rule='original_B512_terminal' if method == 's1' else 'original_max3_terminal',
                                              terminal_trial=old.get('selected_trial_zero_based'),
                                              trace_path=old.get('trace_path'), visible_k=1)
                        else:
                            cs = old['candidates']
                            if method == 'beam20':
                                assert len(cs) == 20 and [c['original_slot_id'] for c in cs] == list(range(20))
                            else:
                                assert len(cs) == 1 and cs[0]['original_slot_id'] == 0
                            c = cs[0]
                            provenance.update(rule='native_model_rank0' if method == 'beam20' else 'original_native_terminal',
                                              original_method=old_method)
                            if method == 'ev':
                                assert c['N_used'] == 400 and c['native_view_only'] is True
                                provenance.update(voting_proposals=400)
                            elif method == 'fewshot':
                                provenance.update(shots=4, num_beams=4, beam_rank=0)
                        endpoint = make_endpoint(model, method, record, c, provenance)
                        ev = endpoint['evaluation']
                        if method == 'sampling_verifier20':
                            assert ev['SR_dir_success'] == old['SR_dir_success']
                        elif method not in ('beam20', 'got'):
                            equality_check(ev, old)
                        elif method == 'got' and short_model == 'mistral':
                            equality_check(ev, old)
                        evaluations.append(ev)
                        output.write(json.dumps(endpoint, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')
                        output_count += 1
                    summary = dict(model_id=model, method_id=method, task_id=task,
                                   status='complete', expected_n=len(expected), completed_n=len(evaluations),
                                   native_k=1, metrics=aggregate_registered(evaluations))
                    if method in ('s1', 'ev', 'fewshot', 'reflexion'):
                        previous = native_summary[short_model][old_method, task]
                        for key in ('n_inputs', 'n_success_dir', 'SR_dir', 'SR_delta', 'Val_input', 'Val_slot'):
                            assert summary['metrics'][key] == previous[key], (model, method, task, key)
                    if method == 'sampling_verifier20':
                        assert summary['metrics']['SR_dir'] == native_summary[short_model][old_method, task]['SR_dir']
                    group_summaries.append(summary)
                    print('COMPLETE', short_model, method, task, len(evaluations), flush=True)
            assert output_count == 3905
            os.replace(temp_path, output_path)
            write_json(ROOT / 'results' / ('cached_' + short_model + '__' + method + '_metrics.json'),
                       [g for g in group_summaries if g['model_id'] == model and g['method_id'] == method])
    expected_count = len(args.models) * len(args.methods) * 3905
    receipt = dict(status='PASS_COMPLETE_CACHED_K1_EXPORT', checked_utc=datetime.now(timezone.utc).isoformat(),
                   wall_seconds=time.monotonic() - started, model_methods=len(args.models) * len(args.methods),
                   groups=len(group_summaries), endpoint_count=expected_count,
                   all_group_denominators_preserved=True, no_new_generation=True, no_new_property_prediction=True,
                   unknown_missing_properties=0, accepted_molecular_exception_occurrences=len(EXCEPTIONS),
                   accepted_molecular_exceptions=EXCEPTIONS, schema_version='BV2-K1-ENDPOINT-v2',
                   metric_version='v2_dir_registered_guarded_legacy_rank_v1',
                   p2_disclosure='Runtime task verifier equals final proxy-property directional criterion. SR@1 equals fixed prefix20 coverage. Actual generation budget remains 40.',
                   output_selection='Actual native outputs; never the old success-selected representative fields.',
                   sources=SOURCES, groups_summary=group_summaries)
    write_json(ROOT / 'reports/cache_export_audit.json', receipt)
    write_json(ROOT / 'results/cached_group_metrics.json', group_summaries)
    print(json.dumps({k: v for k, v in receipt.items() if k not in ('sources', 'groups_summary', 'accepted_molecular_exceptions')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
