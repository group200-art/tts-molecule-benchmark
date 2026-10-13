"""One CPU-only terminal audit/summary/package; no model or property recomputation."""
import csv
import datetime
import json
import os
from pathlib import Path
import sys
import time
import zipfile
from contracts import ROOT,RUN_ID,TASKS,BUDGETS,read,write,sha,manifest,kernel,require_allocation,verify_code,isolated_env
from registered_metrics import aggregate_registered


def mean_available(values):
    values=[v for v in values if v is not None]
    return sum(values)/len(values) if values else None


def group_micro(rows):
    k=kernel();result=k['aggregate'](rows);slots=sum(r['n_slots'] for r in rows)
    result.update(n_property_rows=sum(r['n_property_rows'] for r in rows),n_finite_property_slots=sum(r['n_finite_property_slots'] for r in rows))
    result['property_row_coverage_pct']=100*result['n_property_rows']/slots if slots else None
    result['finite_target_property_coverage_pct']=100*result['n_finite_property_slots']/slots if slots else None
    successes=[r for r in rows if r['SR_dir_success']]
    for prop,label in [('sim','Sim_cached'),('sas','SAS_cached')]:
        vals=[r['selected_property_values'].get(prop) for r in successes]
        result[label]=sum(vals)/len(vals) if vals and all(v is not None for v in vals) else None
    return result


def build_summary():
    all_rows={str(n):{} for n in BUDGETS};summary={};winners=[]
    for task in TASKS:
        expected=[r['input_uid'] for r in manifest(task)]
        for n in BUDGETS:
            rows=[]
            for phase in ('pilot','full'):
                p=ROOT/'cpu'/phase/task/'metrics.json'
                if p.exists():rows.extend(read(p)['by_budget'][str(n)]['inputs'])
            have={r['input_uid'] for r in rows}
            if len(have)!=len(rows) or not have.issubset(expected):raise ValueError('Duplicate/unregistered metric UID')
            rows.sort(key=lambda r:expected.index(r['input_uid']));all_rows[str(n)][task]=rows
            missing=[uid for uid in expected if uid not in have]
            partial=aggregate_registered(rows)
            summary.setdefault(str(n),{})[task]={'expected_inputs':len(expected),'available_inputs':len(rows),'complete':not missing,
                                                'full_denominator_metrics':partial if not missing else None,'partial_metrics':partial,'missing_uids':missing}
        for phase in ('pilot','full'):
            p=ROOT/'cpu'/phase/task/'winners_all.json'
            if p.exists():
                for w in read(p):
                    rep=w['representative']
                    winners.append({k:w[k] for k in ('task_id','input_idx','input_uid','N_used','top_votes','tie_count','class_count','valid_slots','invalid_slots','empty_raw_slots','prefix_cost')})
                    winners[-1].update(candidate_id=rep['candidate_id'],proposal_global_slot=rep['global_slot'],raw_smiles=rep['parsed_candidate'],
                                       canonical_smiles=rep['canonical_smiles'],class_type=rep['class_type'],failure_reason=rep['failure_reason'],
                                       vote_record_path=str(p),tokens_path=rep['tokens_path'],tokens_row=rep['tokens_row'])
    domains={}
    for n in BUDGETS:
        ns=str(n);domains[ns]={}
        for label,tasks in [('IND',TASKS[:5]),('OOD',TASKS[5:]),('ALL',TASKS)]:
            rows=[r for task in tasks for r in all_rows[ns][task]]
            complete=all(summary[ns][t]['complete'] for t in tasks)
            micro=group_micro(rows)
            task_metrics=[summary[ns][t]['partial_metrics'] for t in tasks]
            macro={k:mean_available([m.get(k) for m in task_metrics]) for k in ('SR_dir','SR_delta','Val_input','Val_slot','RI_legacy','RI_abs_eps01','Sim_cached','SAS_cached')}
            macro['conditional_means_available_task_count']={k:sum(m.get(k) is not None for m in task_metrics) for k in ('RI_legacy','RI_abs_eps01','Sim_cached','SAS_cached')}
            domains[ns][label]={'complete':complete,'task_count':len(tasks),'expected_task_inputs':sum(summary[ns][t]['expected_inputs'] for t in tasks),
                               'full_denominator_micro':micro if complete else None,'full_denominator_macro':macro if complete else None,
                               'partial_micro':micro,'partial_macro':macro,'macro_basis':'Equal task weight; undefined conditional means excluded with available-task counts; no zero substitution'}
    return {'run_id':RUN_ID,'tasks':summary,'domains':domains,'complete':all(summary['400'][t]['complete'] for t in TASKS),
            'model_id':'gellmo_p6_mistral','method_id':'p10_chen_vote_ev_native1_v1','native_k':1,'budgets':list(BUDGETS),
            'sampling_realizations':1,'proposal_budget':1562000,'generate_budget':78100,'source_target_reprediction':False},all_rows,winners


def main():
    require_allocation('audit');verify_code();os.environ.update(isolated_env(ROOT/'audit',cpu=True))
    start=time.perf_counter();summary,rows,winners=build_summary()
    write(ROOT/'reports/summary.json',summary)
    with (ROOT/'reports/winners_and_costs.jsonl').open('x') as f:
        for w in winners:f.write(json.dumps(w,ensure_ascii=False,allow_nan=False,separators=(',', ':'))+'\n')
    with (ROOT/'reports/native_metrics.jsonl').open('x') as f:
        for n in BUDGETS:
            for task in TASKS:
                for row in rows[str(n)][task]:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False,separators=(',', ':'))+'\n')
    costs=[]
    for p in sorted((ROOT/'runs').glob('*/costs.json')) if (ROOT/'runs').exists() else []:
        costs.append({'unit':p.parent.name,'path':str(p),'costs':read(p)})
    write(ROOT/'reports/generation_costs.json',{'units':costs,'shared_generation_counted_once':True,
          'total_generate_started':sum(x['costs']['generate_started'] for x in costs),'total_generate_completed':sum(x['costs']['generate_completed'] for x in costs),
          'total_proposals':sum(x['costs']['proposals'] for x in costs),'missing_worker_costs':'Consult calls/started records and Slurm; absence is unknown, never zero',
          'prefix_costs':'See per-output actual full-prefix calls; no Slurm/job overhead prorating'})
    status='COMPLETE_RESULTS_PACKAGED_AUDIT_TERMINAL_PENDING' if summary['complete'] else 'PARTIAL_RESULTS_PACKAGED_AUDIT_TERMINAL_PENDING'
    lines=['# P10 EV 十任务报告 v1','', '执行编号：`'+RUN_ID+'`。状态：`'+status+'`。',
           '', 'P(6)-Mistral，Chen Vote exact molecular identity 适配；主 N400、native K1；单次随机实现。pilot 属于全量，未重复生成。',
           '未完成分母的正式指标为 null，另见 summary.json 的 partial_metrics 与 missing_uids。', '',
           '|任务|已评价/全量|SR_dir %|SR_delta %|Val_slot %|RI_legacy|','|---|---:|---:|---:|---:|---:|']
    def fmt(v):return 'null' if v is None else '{:.6g}'.format(v)
    for task in TASKS:
        s=summary['tasks']['400'][task];m=s['full_denominator_metrics'] or {}
        lines.append('|{}|{}/{}|{}|{}|{}|{}|'.format(task,s['available_inputs'],s['expected_inputs'],*(fmt(m.get(k)) for k in ('SR_dir','SR_delta','Val_slot','RI_legacy'))))
    lines.extend(['','投票结果是获胜等价类最早原槽，不代表400个 proposal 中至少一个成功。无效票、空串和重复均保留。',
                  'IND/OOD 等权 task macro、task-input micro、性质覆盖、APS 和 cached Sim/SAS 见 summary.json。',
                  '共有3905个task-input，不等同3905个独立分子；base upstream revision仍为unknown/null。',
                  '作者shuffle/最后raw代表/共同NNN，与本次原序/最早原parsed代表/normalized-raw失败类的区别见 METHOD_CARD.md。',
                  '所有raw、token IDs、类票数和失败留在E；紧凑包只附原件索引。最终审计作业终态需由本任务实际核实后写包外收据。',''])
    report='\n'.join(lines)
    for name in ('P10_EV_TEN_TASK_REPORT_v1.md','CHECKPOINT_P10_EV_TEN_TASKS_v1.md'):
        with (ROOT/'reports'/name).open('x') as f:f.write(report)
    lines=['# P10 EV 前缀读出 v1','', '四点共享同一N400轨迹；N1与N20都承担完整首个20-return调用。', '',
           '|N_used|最小物理proposals/input|calls/input|全量 SR_dir %|全量 SR_delta %|','|---:|---:|---:|---:|---:|']
    for n in BUDGETS:
        m=summary['domains'][str(n)]['ALL']['full_denominator_micro'] or {}
        lines.append('|{}|{}|{}|{}|{}|'.format(n,max(20,n),max(1,n//20),fmt(m.get('SR_dir')),fmt(m.get('SR_delta'))))
    lines.extend(['','实际prefix调用时间/tokens保存在winners_and_costs.jsonl。共享生成只计一次，完整worker/Slurm成本单列，不按N/400拆账。',''])
    with (ROOT/'reports/P10_EV_PREFIX_READOUT_v1.md').open('x') as f:f.write('\n'.join(lines))
    raw_index=[]
    for base in ('runs','cpu'):
        if (ROOT/base).exists():
            for p in sorted((ROOT/base).rglob('*')):
                if p.is_file() and '/cache/' not in str(p) and '/tmp/' not in str(p):
                    raw_index.append({'path':str(p),'size_bytes':p.stat().st_size,'sha256':sha(p)})
    write(ROOT/'reports/raw_and_science_index.json',raw_index)
    write(ROOT/'reports/audit_summary.json',{'run_id':RUN_ID,'status':status,'expected_outputs':15620,'available_outputs':len(winners),
          'raw_files':len(raw_index),'raw_bytes':sum(x['size_bytes'] for x in raw_index),'wall_seconds_before_packaging':time.perf_counter()-start,
          'no_new_property_computation':True,'no_new_generation':True,'requires_external_terminal_receipt':True})
    # Package only compact source, contracts, reports and scheduler/development receipts.
    members=[]
    for base in ('src','contracts','configs','control','evidence','tests','reports','jobs','logs'):
        for p in sorted((ROOT/base).rglob('*')):
            if p.is_file() and p.suffix not in ('.pyc','.zip') and p.name!='prompts.json':members.append(p)
    path=ROOT/'p10_ev_ten_task_summary_v1.zip'
    with zipfile.ZipFile(str(path),'x',compression=zipfile.ZIP_DEFLATED,allowZip64=True) as z:
        for p in members:z.write(str(p),str(p.relative_to(ROOT)))
    with zipfile.ZipFile(str(path)) as z:
        bad=z.testzip()
        if bad:raise ValueError('ZIP CRC failed: '+bad)
        if len(z.namelist())!=len(members):raise ValueError('ZIP member count mismatch')
    write(ROOT/'zip_receipt.json',{'run_id':RUN_ID,'status':status,'path':str(path),'size_bytes':path.stat().st_size,'sha256':sha(path),
                                 'member_count':len(members),'crc':'PASS','complete_scientific_results':summary['complete'],
                                 'audit_job_id':os.environ['SLURM_JOB_ID'],'audit_terminal_confirmed':False,
                                 'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})


if __name__=='__main__':main()
