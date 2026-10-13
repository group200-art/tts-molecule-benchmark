"""Own-job Slurm accounting; unknown/partial costs are never invented as zero."""
import json
import re
import subprocess
import datetime
from common import ROOT, CONTRACT, EXECUTION_ID, read, write, unit_key, utc

TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}
FIELDS = ['JobID', 'JobIDRaw', 'State', 'ExitCode', 'ElapsedRaw', 'AllocCPUS', 'AllocTRES', 'ReqTRES', 'TimelimitRaw', 'Submit', 'Start', 'End', 'NodeList', 'CPUTimeRAW']


def known_jobs():
    out = {}
    for p in (ROOT / 'control').glob('submitted_*.json'):
        r = read(p)
        if r.get('job_id'):
            out[r['stage']] = r
    return out


def snapshot(label, include_all=True):
    jobs = known_jobs()
    ids = [v['job_id'] for k, v in jobs.items() if include_all or k == 'final_summary']
    result = {'time_utc': utc(), 'jobs': ids, 'rows': [], 'available': False}
    if ids:
        cmd = ['sacct', '-X', '-j', ','.join(ids), '--parsable2', '--noheader', '--format=' + ','.join(f + ('%300' if f in ('AllocTRES', 'ReqTRES') else '%100') for f in FIELDS)]
        result['command'] = cmd
        try:
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=30)
            result.update(returncode=p.returncode, stdout=p.stdout, stderr=p.stderr, available=p.returncode == 0)
            if p.returncode == 0:
                for line in p.stdout.splitlines():
                    cells = line.split('|')
                    assert len(cells) >= len(FIELDS)
                    result['rows'].append(dict(zip(FIELDS, [v.strip() for v in cells[:len(FIELDS)]])))
        except BaseException as err:
            result['error'] = repr(err)
    write(ROOT / 'control' / ('accounting_' + label + '.json'), result, replace=True)
    return result


def terminal(evidence, job_id):
    matches = [r for r in evidence['rows'] if r['JobID'] == job_id]
    return len(matches) == 1 and matches[0]['State'].split()[0].rstrip('+') in TERMINAL


def allocation_totals(evidence, jobs):
    expected = {}
    for stage, info in jobs.items():
        if stage in ('pilot', 'full'):
            for u in CONTRACT['planned_logical_shards']:
                if u['stage'] == stage:
                    expected[info['job_id'] + '_' + str(u['array_index'])] = (stage, u)
        else:
            expected[info['job_id']] = (stage, None)
    byid = {r['JobID']: r for r in evidence['rows'] if r['JobID'] in expected}
    actual, missing, seen_raw = [], [], set()
    for jid, (stage, u) in expected.items():
        r = byid.get(jid)
        if r is None:
            missing.append(jid)
            continue
        assert r['JobIDRaw'] not in seen_raw
        seen_raw.add(r['JobIDRaw'])
        state = r['State'].split()[0].rstrip('+')
        is_terminal = state in TERMINAL
        elapsed = int(r['ElapsedRaw']) if r['ElapsedRaw'].isdigit() else None
        cpu = int(r['AllocCPUS']) if r['AllocCPUS'].isdigit() else None
        tres = dict(field.split('=', 1) for field in r['AllocTRES'].split(',') if '=' in field)
        allocated = bool(tres) and elapsed is not None
        try:
            queued = (datetime.datetime.fromisoformat(r['Start']) - datetime.datetime.fromisoformat(r['Submit'])).total_seconds()
        except (KeyError, ValueError):
            queued = None
        gpu = int(tres.get('gres/gpu', '0')) if allocated else None
        if allocated and stage in ('pilot', 'full'):
            if gpu == 0:
                typed = [int(v) for k, v in tres.items() if k.startswith('gres/gpu:')]
                gpu = typed[0] if len(typed) == 1 else None
            assert gpu in (None, 1)
        cost = {'job_id': jid, 'job_id_raw': r['JobIDRaw'], 'stage': stage, 'state': state, 'exit_code': r['ExitCode'],
                'terminal': is_terminal, 'elapsed_seconds': elapsed, 'allocated_cpus': cpu, 'allocated_gpus': gpu,
                'queue_seconds': queued,
                'gpu_hours': elapsed * gpu / 3600 if allocated and gpu is not None else None,
                'cpu_core_hours': elapsed * cpu / 3600 if allocated and cpu is not None else None,
                'requested_unit_cap_hours': u['gpu_allocation_cap_hours'] if u else None, 'raw_sacct': r}
        actual.append(cost)
    all_terminal = not missing and len(actual) == len(expected) and all(r['terminal'] for r in actual)
    gpurows = [r for r in actual if r['stage'] in ('pilot', 'full')]
    cpur = [r for r in actual if r['stage'] not in ('pilot', 'full')]
    gpu_known = all(r['gpu_hours'] is not None for r in gpurows)
    cpu_known = all(r['cpu_core_hours'] is not None for r in actual)
    return {'source': 'sacct -X; exact scientific array elements and CPU job rows, JobIDRaw counted once; no array-parent/batch/extern addition',
            'all_submitted_allocations_terminal': all_terminal, 'missing_job_ids': missing, 'jobs': actual,
            'actual_gpu_hours': sum(r['gpu_hours'] for r in gpurows) if all_terminal and gpu_known else None,
            'observed_gpu_hours_lower_bound': sum(r['gpu_hours'] for r in gpurows if r['gpu_hours'] is not None),
            'actual_cpu_core_hours': sum(r['cpu_core_hours'] for r in actual) if all_terminal and cpu_known else None,
            'postprocessing_cpu_core_hours': sum(r['cpu_core_hours'] for r in cpur) if all_terminal and all(r['cpu_core_hours'] is not None for r in cpur) else None}


def collect_costs(evidence=None):
    evidence = evidence or snapshot('cost_collection')
    groups = {str(G): {'generate_started_known': 0, 'generate_completed_known': 0, 'prompt_tokens_completed_known': 0,
                       'model_generate_returned_known': 0,
                       'output_tokens_completed_known': 0, 'generate_seconds_measured': 0.0,
                       'property_stage_seconds_excluding_shared_load_measured': 0.0,
                       'property_requests_new_known': 0, 'property_requests_cached_known': 0,
                       'admet_predict_calls_known': 0, 'admet_predict_seconds_measured_nested': 0.0,
                       'admet_predict_completed_known': 0,
                       'standard_property_and_precision_seconds_measured_nested': 0.0,
                       'standalone_G_allocation_gpu_hours': None} for G in (12, 6, 24)}
    shared, unknown, unit_records = [], [], []
    from audit import jsonlines
    for u in CONTRACT['planned_logical_shards']:
        directory = ROOT / 'workers' / unit_key(u)
        if not directory.exists():
            unknown.append({'unit': unit_key(u), 'reason': 'not_started_or_not_submitted; not a zero-cost completed run'})
            continue
        status = read(directory / 'unit_status.json') if (directory / 'unit_status.json').exists() else None
        generation = read(directory / 'generation_cost.json') if (directory / 'generation_cost.json').exists() else None
        props = read(directory / 'property_cost.json') if (directory / 'property_cost.json').exists() else None
        calls_path = directory / 'calls.jsonl'
        if calls_path.exists():
            starts, results = set(), set()
            try:
                for call in jsonlines(calls_path):
                    G = str(call['context']['G'])
                    row = groups[G]
                    if call['event'] == 'started':
                        assert call['call_index'] not in starts
                        starts.add(call['call_index'])
                        row['generate_started_known'] += 1
                    elif call['event'] == 'result':
                        assert call['call_index'] not in results
                        results.add(call['call_index'])
                        row['model_generate_returned_known'] += int(call.get('model_generate_returned', call['completed']))
                        if call['cuda_wall_seconds'] is not None:
                            row['generate_seconds_measured'] += call['cuda_wall_seconds']
                        if call['completed']:
                            row['generate_completed_known'] += 1
                            row['prompt_tokens_completed_known'] += call['prompt_tokens']
                            row['output_tokens_completed_known'] += call['tokens_through_first_eos']
                if starts != results:
                    unknown.append({'unit': unit_key(u), 'reason': 'started call lacks terminal record', 'call_indices': sorted(starts - results)})
            except BaseException as err:
                unknown.append({'unit': unit_key(u), 'record': 'calls.jsonl', 'reason': repr(err), 'basis': 'valid recorded prefix only'})
        elif generation is not None:
            for G, values in generation['by_G'].items():
                row = groups[G]
                row['generate_started_known'] += values['calls_started']
                row['generate_completed_known'] += values['calls_completed']
                row['prompt_tokens_completed_known'] += values['prompt_tokens']
                row['output_tokens_completed_known'] += values['output_tokens_through_eos']
                row['generate_seconds_measured'] += values['generate_seconds']
        if props is not None:
            for G, values in props['by_G'].items():
                row = groups[G]
                row['property_stage_seconds_excluding_shared_load_measured'] += values['stage_seconds_excluding_shared_load']
                row['property_requests_new_known'] += values['requests_new']
                row['property_requests_cached_known'] += values['requests_cached']
        for filename, kind in [('inf_predictions.jsonl', 'admet'), ('eval_tool_rows.jsonl', 'standard')]:
            path = directory / filename
            if path.exists():
                try:
                    for entry in jsonlines(path):
                        G = str(entry['context']['G'])
                        if kind == 'admet' and entry['event'] == 'result':
                            groups[G]['admet_predict_calls_known'] += 1
                            groups[G]['admet_predict_completed_known'] += int(entry['completed'])
                            groups[G]['admet_predict_seconds_measured_nested'] += entry['predict_wall_seconds']
                        elif kind == 'standard':
                            groups[G]['standard_property_and_precision_seconds_measured_nested'] += entry['row']['standard_property_and_precision_seconds'] - entry['row']['shared_standard_load_seconds']
                except BaseException as err:
                    unknown.append({'unit': unit_key(u), 'record': filename, 'reason': repr(err)})
        shared.append({'unit': unit_key(u), 'beneficiary_G': [12, 6, 24],
                       'generation_model_load': {k: v for k, v in (generation or {}).items() if k.startswith('model_load')},
                       'tokenizer_load_seconds': (generation or {}).get('tokenizer_load_seconds'),
                       'generation_initialization_seconds_inclusive': (generation or {}).get('generation_initialization_seconds'),
                       'property_startup_seconds': props.get('startup_seconds') if props else None,
                       'admet_load_seconds': props.get('admet_shared_and_nested_cost', {}).get('load_wall_seconds') if props else None,
                       'admet_internal_load_counts': props.get('admet_shared_and_nested_cost', {}) if props else None,
                       'standard_shared_load_seconds': props.get('standard_shared_load_seconds') if props else None,
                       'worker_wall_seconds': status.get('wall_seconds') if status else None,
                       'shared_startup_io_unattributed_seconds': None, 'note': 'shared records once; worker/process/predict timing is nested, not additive'})
        load_result = directory / 'inf_load_result.json'
        if load_result.exists():
            detail = read(load_result)
            shared[-1]['admet_load_seconds'] = detail['cost']['load_wall_seconds']
            shared[-1]['admet_shared_setup_seconds_inclusive'] = detail['cost']['shared_setup_wall_seconds']
            shared[-1]['admet_internal_load_counts'] = detail['cost']
        if not status or status['status'] != 'complete':
            unknown.append({'unit': unit_key(u), 'reason': 'partial worker; unmeasured calls/tokens/durations remain unknown; recorded amounts are lower bounds'})
        unit_records.append({'unit': unit_key(u), 'status': status, 'generation': generation, 'property': props})
    analysis = read(ROOT / 'results/analysis_cost.json') if (ROOT / 'results/analysis_cost.json').exists() else None
    result = {'execution_id': EXECUTION_ID, 'time_utc': utc(), 'planned_calls': 351450, 'planned_output_token_cap': 74038800,
              'approved_gpu_allocation_cap_hours': 630, 'by_G': groups, 'shared_once': shared,
              'analysis_measurements': analysis, 'allocation': allocation_totals(evidence, known_jobs()),
              'unknown_or_partial': unknown, 'unit_records': unit_records,
              'K_cost_rule': 'K1/5/10/20 inherit full graph cost; no K-proportional allocation',
              'do_not_sum': 'generation, property RPC, internal predictor, worker wall, and Slurm allocation are nested clocks',
              'standalone_G_allocation': 'unknown; no arbitrary split of shared loads or Slurm allocations'}
    write(ROOT / 'results/costs.json', result, replace=True)
    return result


def close_accounting():
    final = known_jobs()['final_summary']['job_id']
    evidence = snapshot('terminal')
    assert terminal(evidence, final)
    costs = collect_costs(evidence)
    if not costs['allocation']['all_submitted_allocations_terminal']:
        raise RuntimeError('Final CPU ended but some submitted allocation terminal records are missing; do not invent cost')
    analysis = read(ROOT / 'control/final_analysis_finished.json') if (ROOT / 'control/final_analysis_finished.json').exists() else None
    status = analysis['scientific_status'] if analysis else 'partial'
    closing = {'execution_id': EXECUTION_ID, 'time_utc': utc(), 'scientific_status': status,
               'authorization_closed': True, 'science_reexecuted_for_accounting': False,
               'final_summary_job_id': final, 'accounting_available': costs['allocation']['actual_gpu_hours'] is not None}
    write(ROOT / 'control/authorization_closed.json', closing)
    paragraph = '\n\nTerminal accounting completed at {}. Scientific status: {}. All submitted allocations are terminal. GPU allocation: {} GPUh; total CPU: {} coreh. Authorization closed; no further scientific execution. Metrics were not rerun for this accounting update.\n'.format(
                    closing['time_utc'], status, costs['allocation']['actual_gpu_hours'], costs['allocation']['actual_cpu_core_hours'])
    for name in ['SUMMARY_P8_GWIDTH_v1.md', 'CHECKPOINT_P8_GWIDTH_TEN_TASKS_v1.md']:
        path = ROOT / 'reports' / name
        previous = path.read_text() if path.exists() else '# P8 partial result\n\nFinal analysis did not complete. Original trajectories and failures remain in workers and logs; full metrics are unavailable.\n'
        previous = previous.replace('Terminal accounting pending; no further scientific submissions.', 'Terminal accounting complete; authorization closed.')
        previous = previous.replace('Final CPU allocation is pending until external terminal accounting.', 'Final CPU allocation is included in the terminal accounting below.')
        path.write_text(previous + paragraph)
    from summarize import package
    receipt = package()
    write(ROOT / 'control/accounting_watcher_finished.json', {'time_utc': utc(), 'zip_sha256': receipt['sha256'], 'status': status})
