"""Once-only Slurm stage control. Pilot PASS authorizes exactly one full array."""
import os
import re
import subprocess
import sys
import time
import traceback
from common import ROOT, CONTRACT, EXECUTION_ID, read, write, append, unit_key, verify_execution, require_allocation, configure_environment, utc


def jobs():
    out = {}
    for path in (ROOT / 'control').glob('submitted_*.json'):
        r = read(path)
        if r.get('job_id'):
            out[r['stage']] = r
    return out


def submit(stage, dependency=None):
    assert stage in ('pilot', 'pilot_audit', 'full', 'final_summary')
    verify_execution()
    if stage == 'full':
        audit = read(ROOT / 'control/pilot_audit.json')
        assert audit['pass'] is True and len(audit['units']) == 2
    intent = {'execution_id': EXECUTION_ID, 'stage': stage, 'dependency': dependency, 'time_utc': utc(), 'attempt': 1}
    # An uncertain submission is never retried. This exclusive intent is permanent evidence.
    write(ROOT / 'control' / ('submission_intent_' + stage + '.json'), intent)
    cmd = ['sbatch', '--parsable']
    if dependency:
        assert re.fullmatch(r'afterany:[0-9]+', dependency)
        cmd.append('--dependency=' + dependency)
    cmd.append(str(ROOT / 'code' / (stage + '.sbatch')))
    result = dict(intent, command=cmd, job_id=None)
    try:
        submit_env = dict(os.environ)
        submit_env.pop('CUDA_VISIBLE_DEVICES', None)
        submit_env = {k: v for k, v in submit_env.items() if not k.startswith('SBATCH_')}
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=45, cwd=str(ROOT), env=submit_env)
        result.update(returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
        if p.returncode != 0:
            raise RuntimeError('Slurm rejected submission: ' + p.stderr)
        match = re.fullmatch(r'([0-9]+)(;[^\s]+)?\s*', p.stdout)
        if not match:
            raise RuntimeError('Uncertain sbatch response; no retry')
        result['job_id'] = match.group(1)
        if stage in ('pilot', 'full'):
            result['mapping'] = [dict(u, execution_approved=True, historical_design_execution_approved=u['execution_approved'],
                                      slurm_array_element=result['job_id'] + '_' + str(u['array_index']), unit_key=unit_key(u))
                                 for u in CONTRACT['planned_logical_shards'] if u['stage'] == stage]
    except BaseException as err:
        result.update(exception=repr(err), traceback=traceback.format_exc())
        raise
    finally:
        write(ROOT / 'control' / ('submitted_' + stage + '.json'), result)
    refresh_submission_checkpoint()
    return result['job_id']


def refresh_submission_checkpoint():
    mapping = jobs()
    text = '# P8 execution checkpoint\n\nExecution: ' + EXECUTION_ID + '\n\n'
    text += 'Status: submitted/running; scientific completion is not inferred from queue submission.\n\n'
    text += 'Main: G12-N28-K20. G6/G24 are independent generation budgets; all same-graph K analyses reuse the full graph.\n\n'
    text += '| Stage | JobID | Dependency | Map |\n|---|---|---|---|\n'
    for stage in ('pilot', 'pilot_audit', 'full', 'final_summary'):
        r = mapping.get(stage)
        if r:
            text += '| {} | {} | {} | {} |\n'.format(stage, r['job_id'], r['dependency'],
                         'control/submitted_' + stage + '.json')
    text += '\nPilot mechanical PASS submits the full array automatically. No SR/validity/diversity gate, retries, or extra groups.\n'
    text += '\nExact logical map and two-wave seeds: configs/design_contract.json planned_logical_shards. Pilot10 UID are excluded from full157.\n'
    text += '\nMaximum requested allocation: pilot2h + full628h = 630 GPUh; CPU audit1h and final2h, each4CPU/16GiB.\n'
    text += '\nFinal reports, receipt, and archive are produced by the single final-summary job; external accounting appends its terminal allocation without rerunning metrics.\n'
    (ROOT / 'reports/CHECKPOINT_P8_GWIDTH_TEN_TASKS_v1.md').write_text(text)


def run_audit():
    require_allocation()
    from audit import pilot_audit
    result = pilot_audit()
    full_id = None
    if result['pass']:
        try:
            full_id = submit('full')
        except BaseException as err:
            write(ROOT / 'control/full_submission_failure.json', {'time_utc': utc(), 'exception': repr(err), 'retry': False})
    dep = full_id or os.environ['SLURM_JOB_ID']
    submit('final_summary', 'afterany:' + dep)
    if not result['pass']:
        raise RuntimeError('Pilot mechanical FAIL; full was not submitted')


def watch_accounting():
    from accounting import snapshot, terminal, close_accounting
    write(ROOT / 'control/accounting_watcher_started.json', {'pid': os.getpid(), 'time_utc': utc(), 'purpose': 'own-stage terminal accounting only; no scientific reruns'})
    while True:
        mapping = jobs()
        final = mapping.get('final_summary')
        if final:
            evidence = snapshot('watch', include_all=False)
            if terminal(evidence, final['job_id']):
                try:
                    close_accounting()
                    return
                except RuntimeError as err:
                    # Accounting propagation may lag termination; no scientific or sbatch retry.
                    append(ROOT / 'control/accounting_lag.jsonl', {'time_utc': utc(), 'error': repr(err)})
        time.sleep(60)


if __name__ == '__main__':
    configure_environment(ROOT / 'cache' / ('control-' + str(os.getpid())), cpu=True)
    action = sys.argv[1]
    if action == 'submit-pilot':
        checks = read(ROOT / 'control/cpu_checks.json')
        assert checks['pass'] is True
        assert read(ROOT / 'control/resource_check.json')['pass'] is True
        pilot = submit('pilot')
        audit = submit('pilot_audit', 'afterany:' + pilot)
        print('pilot={} pilot_audit={}'.format(pilot, audit), flush=True)
    elif action == 'audit':
        run_audit()
    elif action == 'final':
        require_allocation()
        verify_execution()
        from summarize import main
        try:
            main()
        except BaseException as err:
            write(ROOT / 'control/final_analysis_failure.json', {'time_utc': utc(), 'exception': repr(err), 'traceback': traceback.format_exc(), 'retry': False})
            raise
    elif action == 'watch-accounting':
        watch_accounting()
    else:
        raise ValueError('Unknown stage')
