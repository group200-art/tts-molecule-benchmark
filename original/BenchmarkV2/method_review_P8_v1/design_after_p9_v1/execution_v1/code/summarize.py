"""One cached-record-only final analysis; partial tasks keep full metrics null."""
import collections
import gzip
import json
import os
import time
import traceback
import zipfile
from common import ROOT, CONTRACT, EXECUTION_ID, read, write, sha, unit_key, manifest, kernel, utc
from metrics import evaluate_registered, aggregate_registered
from audit import validate_graph


def output_candidates(graph, K):
    nodes = {n['node_id']: n for n in graph['nodes']}
    out = []
    for rank, nid in enumerate(graph['top20'][:K]):
        r, n = graph['properties'][nid], nodes[nid]
        out.append({'candidate_id': nid, 'original_slot_id': rank, 'created_order': n['created_order'],
                    'raw_smiles': r['request_smiles'], 'rdkit_valid': r['rdkit_valid'],
                    'properties': r['properties'], 'property_row_exists': r['property_row_exists']})
    return out


def mixed_aggregate(rows):
    out = kernel()['aggregate'](rows)
    successes = [r for r in rows if r['SR_dir_success']]
    out['n_success_delta'] = sum(r['SR_delta_success'] for r in rows)
    for p, label in [('sim', 'Sim_cached'), ('sas', 'SAS_cached')]:
        values = [r['selected_property_values'].get(p) for r in successes]
        out[label + '_available_count'] = sum(v is not None for v in values)
        out[label] = sum(values) / len(values) if values and all(v is not None for v in values) else None
    names = sorted({p for r in rows for p in r['source_properties']})
    out['APS'] = {}
    for p in names:
        values = [r['selected_property_values'][p] for r in successes if p in r['source_properties']]
        out['APS'][p] = sum(values) / len(values) if values else None
    total = sum(r['n_slots'] for r in rows)
    out.update(n_property_rows=sum(r['n_property_rows'] for r in rows), n_finite_property_slots=sum(r['n_finite_property_slots'] for r in rows))
    out['property_row_coverage_pct'] = 100 * out['n_property_rows'] / total if total else None
    out['finite_target_property_coverage_pct'] = 100 * out['n_finite_property_slots'] / total if total else None
    return out


def macro(metrics, tasks):
    labels = ['SR_dir', 'SR_delta', 'Val_input', 'Val_slot', 'property_row_coverage_pct',
              'finite_target_property_coverage_pct', 'RI_legacy', 'RI_abs_eps01', 'Sim_cached', 'SAS_cached']
    out = {'n_tasks': len(tasks), 'available_task_counts': {}, 'APS': {}}
    for label in labels:
        values = [metrics[t['task_id']].get(label) for t in tasks]
        out['available_task_counts'][label] = sum(v is not None for v in values)
        out[label] = sum(values) / len(values) if values and all(v is not None for v in values) else None
    for p in sorted({o['property'] for t in tasks for o in t['objectives']}):
        values = [metrics[t['task_id']]['APS'].get(p) for t in tasks if p in [o['property'] for o in t['objectives']]]
        out['APS'][p] = {'value': sum(values) / len(values) if values and all(v is not None for v in values) else None,
                         'available_tasks': sum(v is not None for v in values), 'eligible_tasks': len(values)}
    return out


def diagnostic(graph):
    nodes, props, source = graph['nodes'], graph['properties'], graph['record']['source_properties']
    names = list(source)
    counts = collections.Counter(r['inf_canonical'] for r in props.values() if r['inf_canonical'])
    byid = {n['node_id']: n for n in nodes}
    operations = {}
    for operation in ['Generate', 'Refine', 'Merge', 'Aggregate']:
        subset = [n for n in nodes if n['operation'] == operation]
        row = {'n': len(subset), 'empty': 0, 'valid': 0, 'duplicate_in_full_pool': 0, 'success_dir': 0, 'success_delta': 0, 'child_equals_any_parent': 0}
        thresholds = {o['property']: o['supplementary_success_delta'] for t in CONTRACT['tasks'] if t['task_id'] == graph['record']['task_id'] for o in t['objectives']}
        for n in subset:
            r = props[n['node_id']]
            row['empty'] += int(not n['parsed_smiles'])
            row['valid'] += int(r['rdkit_valid'])
            row['duplicate_in_full_pool'] += int(bool(r['inf_canonical']) and counts[r['inf_canonical']] > 1)
            eligible = r['rdkit_valid'] and kernel()['finite_values'](r['properties'], names)
            row['success_dir'] += int(eligible and kernel()['strictly_improves'](source, r['properties'], names))
            row['success_delta'] += int(eligible and kernel()['meets_delta'](source, r['properties'], names, thresholds))
            row['child_equals_any_parent'] += int(any(r['request_smiles'] == props[p]['request_smiles'] for p in n['parents']))
        operations[operation] = row
    def state(nid):
        p = props[nid]
        return (p['rdkit_valid'], p['request_smiles'])
    parent_states = [state(p) for p in graph['parents']]
    return {'task_id': graph['record']['task_id'], 'input_uid': graph['record']['input_uid'], 'G': graph['G'],
            'operations': operations, 'top20_operations': dict(collections.Counter(byid[n]['operation'] for n in graph['top20'])),
            'unique_parent_states': len(set(parent_states)), 'parent_states_ordered': parent_states,
            'top20_states_ordered': [state(n) for n in graph['top20']],
            'multi_parent_distinct_counts': [{'node': n['local_node_id'], 'nominal': len(n['parents']),
                                            'distinct': len({state(p) for p in n['parents']})} for n in nodes if len(n['parents']) > 1],
            'wave0_top20_present_count': sum(n.startswith('g') for n in [byid[n]['local_node_id'] for n in graph['top20']]),
            'wave0_valid_count': sum(props[n['node_id']]['rdkit_valid'] for n in nodes[:graph['G']]),
            'full_valid_count': sum(r['rdkit_valid'] for r in props.values()),
            'wave0_success_dir_any': operations['Generate']['success_dir'] > 0,
            'full_success_dir_any': any(r['success_dir'] > 0 for r in operations.values())}


def artifact_index():
    path = ROOT / 'results/large_artifact_index.jsonl.gz'
    temp = ROOT / 'tmp/artifact_index.jsonl.gz'
    count, total = 0, 0
    with gzip.open(str(temp), 'wt', encoding='utf-8') as f:
        walks = (entry for sub in ['workers', 'logs'] for entry in os.walk(str(ROOT / sub)))
        for base, dirs, names in walks:
            dirs[:] = sorted(d for d in dirs if d not in ('cache', '__pycache__'))
            for name in sorted(names):
                p = ROOT / base / name
                if p.is_symlink():
                    raise ValueError('Unexpected symlink in worker artifacts')
                item = {'path': str(p), 'relative_path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size, 'sha256': sha(p),
                        'mutable_at_index': p.parent == ROOT / 'logs' and ('final_summary_' in p.name or 'accounting_watcher' in p.name)}
                f.write(json.dumps(item) + '\n')
                count += 1
                total += item['bytes']
    os.replace(str(temp), str(path))
    write(ROOT / 'results/large_artifact_totals.json', {'files': count, 'bytes': total, 'index': str(path),
                    'excluded': 'model/environment assets, transient per-worker cache directories; raw scientific trajectories retained'}, replace=True)


def package():
    target = ROOT / 'p8_gwidth_ten_task_summary_v1.zip'
    temp = ROOT / 'tmp' / ('review_pack_' + str(os.getpid()) + '.zip')
    members = []
    with zipfile.ZipFile(str(temp), 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for sub in ['code', 'configs', 'vendor', 'sources', 'tests', 'control', 'reports', 'results']:
            for base, dirs, names in os.walk(str(ROOT / sub)):
                dirs[:] = sorted(d for d in dirs if d != '__pycache__' and not d.endswith(('_fixture', '_cache')))
                for name in sorted(names):
                    if '.tmp.' in name or name == 'accounting_watch.json':
                        continue
                    p = Path(base) / name
                    if p.is_symlink():
                        raise ValueError('Unexpected review symlink')
                    data = p.read_bytes()
                    relative = str(p.relative_to(ROOT))
                    z.writestr(relative, data)
                    members.append({'path': relative, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    with zipfile.ZipFile(str(temp)) as z:
        assert z.testzip() is None and len(z.infolist()) == len(members)
        for member in members:
            data = z.read(member['path'])
            assert len(data) == member['bytes'] and hashlib.sha256(data).hexdigest() == member['sha256']
    os.replace(str(temp), str(target))
    receipt = {'execution_id': EXECUTION_ID, 'time_utc': utc(), 'path': str(target), 'bytes': target.stat().st_size,
               'sha256': sha(target), 'member_count': len(members), 'crc_and_member_hashes_verified': True, 'members': members,
               'external_receipt_not_in_archive': True, 'cost_terminal': (ROOT / 'control/authorization_closed.json').exists()}
    write(ROOT / 'zip_receipt.json', receipt, replace=True)
    return receipt


def main():
    from accounting import collect_costs, snapshot
    write(ROOT / 'control/final_analysis_started.json', {'time_utc': utc(), 'job_id': os.environ.get('SLURM_JOB_ID'), 'attempt': 1})
    start = time.perf_counter()
    by_task, graphs_diag, missing, errors = {}, {}, [], []
    unit_states = []
    measured = {'score_seconds': {}, 'parent_selection_seconds': {}, 'metrics_seconds': {}, 'score_wave0_seconds': {}, 'score_full_seconds': {}, 'final_selection_seconds': {}}
    perinput = gzip.open(str(ROOT / 'results/per_input_topK.jsonl.gz'), 'wt', encoding='utf-8')
    diagfile = gzip.open(str(ROOT / 'results/per_input_graph_diagnostics.jsonl.gz'), 'wt', encoding='utf-8')
    try:
        for u in CONTRACT['planned_logical_shards']:
            statuspath = ROOT / 'workers' / unit_key(u) / 'unit_status.json'
            if not statuspath.exists():
                unit_states.append({'unit': unit_key(u), 'status': 'not_started_or_not_submitted'})
                continue
            status = read(statuspath)
            unit_states.append({'unit': unit_key(u), 'status': status['status'], 'exception': status.get('exception'), 'close_errors': status.get('close_errors')})
            for entry in status.get('completed_graphs', []):
                try:
                    graph = read(ROOT / entry['graph'])
                    validate_graph(graph)
                    G, r = graph['G'], graph['record']
                    key = str(G)
                    measured['score_seconds'][key] = measured['score_seconds'].get(key, 0.0) + graph['scoring_seconds']
                    measured['parent_selection_seconds'][key] = measured['parent_selection_seconds'].get(key, 0.0) + sum(e['inclusive_wall_seconds'] for e in graph['execution_events'] if e['role'] == 'select_wave0')
                    measured['final_selection_seconds'][key] = measured['final_selection_seconds'].get(key, 0.0) + sum(e['inclusive_wall_seconds'] for e in graph['execution_events'] if e['role'] == 'select_full')
                    for stage in ('wave0', 'full'):
                        label = 'score_' + stage + '_seconds'
                        measured[label][key] = measured[label].get(key, 0.0) + graph['scoring_seconds_by_stage'][stage]
                    for K in (1, 5, 10, 20):
                        t = time.perf_counter()
                        result = evaluate_registered(r['task_id'], r['input_uid'], r['source_properties'], output_candidates(graph, K), K)
                        result.update(G=G, K=K, source_molecule_uid=r['source_molecule_uid'], input_idx=r['input_idx'],
                                      upstream_cost='full G+16 graph; no extra generation/property inference for K')
                        measured['metrics_seconds'][key] = measured['metrics_seconds'].get(key, 0.0) + time.perf_counter() - t
                        # Keep IDs/values in compact per-UID output; all raw slots remain in indexed graphs.
                        result.pop('candidates')
                        result.pop('slots')
                        by_task.setdefault((G, K, r['task_id']), []).append(result)
                        perinput.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
                    d = diagnostic(graph)
                    graphs_diag[(G, r['input_uid'])] = d
                    diagfile.write(json.dumps(d, ensure_ascii=False, allow_nan=False) + '\n')
                except BaseException as err:
                    errors.append({'graph': entry, 'exception': repr(err), 'traceback': traceback.format_exc()})
    finally:
        perinput.close()
        diagfile.close()
    tasks, domains = [], []
    for G in (12, 6, 24):
        for K in (1, 5, 10, 20):
            task_metrics = {}
            complete_tasks = []
            for task in CONTRACT['tasks']:
                tid = task['task_id']
                rows = by_task.get((G, K, tid), [])
                actual = {r['input_uid'] for r in rows}
                assert len(actual) == len(rows)
                expected = manifest(tid)
                absent = [r['input_uid'] for r in expected if r['input_uid'] not in actual]
                assert actual.issubset({r['input_uid'] for r in expected})
                subset = aggregate_registered(rows)
                full = subset if not absent else None
                if full is not None:
                    complete_tasks.append(task)
                task_metrics[tid] = subset
                tasks.append({'G': G, 'K': K, 'task_id': tid, 'domain': task['domain'], 'expected_n': task['actual_n'],
                              'completed_n': len(rows), 'status': 'complete' if full is not None else 'partial',
                              'full_task_metrics': full, 'completed_subset_metrics': subset,
                              'upstream_cost_basis': 'same full graph for all K; independent deployment allocation unknown'})
                if K == 20:
                    missing.append({'G': G, 'task_id': tid, 'missing_input_uids': absent})
            for domain in ('IND', 'OOD', 'ALL'):
                wanted = [t for t in CONTRACT['tasks'] if domain == 'ALL' or t['domain'] == domain]
                complete = all(t in complete_tasks for t in wanted)
                subset_tasks = [t for t in wanted if t in complete_tasks]
                rows = [r for t in wanted for r in by_task.get((G, K, t['task_id']), [])]
                subset_rows = [r for t in subset_tasks for r in by_task.get((G, K, t['task_id']), [])]
                domains.append({'G': G, 'K': K, 'domain': domain, 'status': 'complete' if complete else 'partial',
                    'full_macro': macro(task_metrics, wanted) if complete else None,
                    'full_micro': mixed_aggregate(rows) if complete else None,
                    'completed_tasks_subset': [t['task_id'] for t in subset_tasks],
                    'completed_tasks_macro': macro(task_metrics, subset_tasks),
                    'completed_tasks_micro': mixed_aggregate(subset_rows), 'all_observed_inputs_micro': mixed_aggregate(rows)})
    write(ROOT / 'results/task_metrics.json', tasks)
    write(ROOT / 'results/domain_metrics.json', domains)
    write(ROOT / 'results/missing_UIDs.json', missing)
    write(ROOT / 'results/analysis_errors.json', errors)
    write(ROOT / 'results/unit_states.json', unit_states)
    write(ROOT / 'results/analysis_cost.json', dict(measured, analysis_wall_seconds_before_index=time.perf_counter() - start,
                         nested_times_do_not_add_to_allocation=True))
    paired = []
    for lo, hi in [(6, 12), (12, 24)]:
        for task in CONTRACT['tasks']:
            lower = {r['input_uid']: r for r in by_task.get((lo, 20, task['task_id']), [])}
            upper = {r['input_uid']: r for r in by_task.get((hi, 20, task['task_id']), [])}
            common = sorted(set(lower) & set(upper))
            row = {'lower_G': lo, 'upper_G': hi, 'task_id': task['task_id'], 'paired_n': len(common), 'expected_n': task['actual_n']}
            for metric in ('dir', 'delta'):
                transition = collections.Counter(str(int(lower[uid]['SR_' + metric + '_success'])) + '->' + str(int(upper[uid]['SR_' + metric + '_success'])) for uid in common)
                row[metric] = {'transitions': dict(transition), 'net_success_count': transition['0->1'] - transition['1->0']}
            same = collections.Counter()
            for uid in common:
                a, b = graphs_diag[(lo, uid)], graphs_diag[(hi, uid)]
                same['parent_ordered_equal'] += int(a['parent_states_ordered'] == b['parent_states_ordered'])
                same['parent_multiset_equal'] += int(sorted(a['parent_states_ordered']) == sorted(b['parent_states_ordered']))
                same['top20_ordered_equal'] += int(a['top20_states_ordered'] == b['top20_states_ordered'])
                same['top20_multiset_equal'] += int(sorted(a['top20_states_ordered']) == sorted(b['top20_states_ordered']))
            row['state_equality_counts'] = dict(same)
            row['equality_identity'] = '(inf_valid, request_smiles), retains invalid and empty states and multiplicity'
            row['state_equality_rates'] = {k: v / len(common) if common else None for k, v in same.items()}
            paired.append(row)
    write(ROOT / 'results/paired_adjacent_G.json', paired)
    grouped_diagnostics = []
    for G in (12, 6, 24):
        for task in CONTRACT['tasks']:
            selected = [d for (g, uid), d in graphs_diag.items() if g == G and d['task_id'] == task['task_id']]
            operations = {}
            for op in ('Generate', 'Refine', 'Merge', 'Aggregate'):
                summed = collections.Counter()
                for d in selected:
                    summed.update(d['operations'][op])
                operations[op] = {'counts': dict(summed), 'rates': {k: v / summed['n'] if summed['n'] else None for k, v in summed.items() if k != 'n'}}
            row = {'G': G, 'task_id': task['task_id'], 'complete_graphs': len(selected), 'operations': operations,
                   'scope': 'completed graph descriptive diagnostics; no causal operator comparison'}
            for field in ('unique_parent_states', 'wave0_top20_present_count', 'wave0_valid_count', 'full_valid_count'):
                row[field + '_histogram'] = dict(collections.Counter(d[field] for d in selected))
            row['wave0_to_full_success_dir'] = dict(collections.Counter(str(int(d['wave0_success_dir_any'])) + '->' + str(int(d['full_success_dir_any'])) for d in selected))
            row['multi_parent_distinct_histogram'] = dict(collections.Counter(str(x['nominal']) + ':' + str(x['distinct']) for d in selected for x in d['multi_parent_distinct_counts']))
            total = collections.Counter()
            for d in selected:
                total.update(d['top20_operations'])
            row['top20_operations'] = dict(total)
            grouped_diagnostics.append(row)
    write(ROOT / 'results/graph_diagnostics_summary.json', grouped_diagnostics)
    snapshot('pre_summary_end')
    costs = collect_costs()
    write(ROOT / 'results/budget_curve_K20.json', {'main_G': 12, 'points': [{'G': G, 'N': G + 16, 'K': 20,
          'is_main': G == 12, 'generation_cost': costs['by_G'][str(G)], 'shared_cost_reference': 'costs.json shared_once',
          'task_metrics': [t for t in tasks if t['G'] == G and t['K'] == 20],
          'domain_metrics': [d for d in domains if d['G'] == G and d['K'] == 20]} for G in (6, 12, 24)]})
    complete = all(not m['missing_input_uids'] for m in missing) and not errors and all(u['status'] == 'complete' for u in unit_states)
    document = '# P8 G-width ten-task results\n\nExecution: ' + EXECUTION_ID + '\n\n'
    document += '**Scientific status: {}.** Main configuration remains G12-N28-K20.\n\n'.format('complete' if complete else 'partial')
    document += 'Three complete generation budgets share one approved method/scorer; K1/5/10/20 are same-graph output prefixes. No additional generation or property inference was used for analysis.\n\n'
    document += '| G | Task | Completed / expected | SR_dir K20 | SR_delta K20 |\n|---|---|---|---|---|\n'
    for t in tasks:
        if t['K'] != 20:
            continue
        m = t['full_task_metrics']
        document += '| {} | {} | {}/{} | {} | {} |\n'.format(t['G'], t['task_id'], t['completed_n'], t['expected_n'],
                '{:.4f}%'.format(m['SR_dir']) if m else 'null', '{:.4f}%'.format(m['SR_delta']) if m else 'null')
    document += '\nTask and domain JSONs contain all four K, full and explicitly labelled subset results, counts, validity, property coverage, conditional RI/APS/Sim/SAS. Missing UIDs are not chemical failures.\n'
    document += '\nAdjacent-G comparisons report matched success gains/losses and parent/output equality. Operation diagnostics are descriptive; no unapproved control or causal operator claim. Source-molecule UID is retained in per-input records.\n'
    document += '\nCosts: results/costs.json; shared assembly is counted once, nested timing is not added to allocation, standalone-G allocation is unknown. Every K inherits full upstream graph cost. Final CPU allocation is pending until external terminal accounting.\n'
    document += '\nRaw trajectories: results/large_artifact_index.jsonl.gz (absolute paths, bytes, SHA256); compact archive excludes models, environments, caches, and large raw trajectories.\n'
    (ROOT / 'reports/SUMMARY_P8_GWIDTH_v1.md').write_text(document)
    from stages import jobs
    checkpoint = '# P8 G-width checkpoint\n\nExecution: ' + EXECUTION_ID + '\n\n'
    checkpoint += 'Scientific status: {}. Terminal accounting pending; no further scientific submissions.\n\n'.format('complete' if complete else 'partial')
    checkpoint += '| Stage | JobID | Dependency |\n|---|---|---|\n'
    for stage, job in jobs().items():
        checkpoint += '| {} | {} | {} |\n'.format(stage, job['job_id'], job['dependency'])
    checkpoint += '\nExact stage/index/task/logical-shard/RNG mapping is in control/submitted_pilot.json and, if submitted, control/submitted_full.json. Pilot results are included once; no retries or extra groups.\n'
    checkpoint += '\nFinal reports: reports/SUMMARY_P8_GWIDTH_v1.md; results/task_metrics.json; results/domain_metrics.json; results/costs.json; zip_receipt.json.\n'
    (ROOT / 'reports/CHECKPOINT_P8_GWIDTH_TEN_TASKS_v1.md').write_text(checkpoint)
    artifact_index()
    write(ROOT / 'control/final_analysis_finished.json', {'time_utc': utc(), 'scientific_status': 'complete' if complete else 'partial',
          'wall_seconds': time.perf_counter() - start, 'completed_graphs': len(graphs_diag), 'metrics_rerun': False})
    package()


from pathlib import Path
import hashlib
