"""Independent completed-result checks; standard library, no property inference."""
import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TASKS = ['BDP','BDQ','BPQ','DPQ','BDPQ','MPQ','BDMQ','BHMQ','BMPQ','HMPQ']
COUNTS = dict(zip(TASKS,[500]*7+[118,191,96]))


def readlines(path):
    with gzip.open(path,'rt') as f:
        return [json.loads(s) for s in f if s.strip()]


def close(a,b):
    assert math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-10),(a,b)


def mean(xs):
    xs=list(xs)
    return sum(xs)/len(xs)


def main():
    novelty_reference=set((ROOT.parents[1]/'data/unique_mols_in_one_ds_pairs.txt').read_text().splitlines())
    audit=json.loads((ROOT/'results/audit.json').read_text())
    assert audit['status']=='PASS_COMPLETE_TWO_MODELS_FULL3905_N20_R20_K20'
    rows=readlines(ROOT/'results/per_input.jsonl.gz')
    pairs=readlines(ROOT/'results/paired_outcomes.jsonl.gz')
    assert len(rows)==23430 and len(pairs)==7810
    keyed={(r['model_id'],r['task_id'],r['input_uid'],r['readout']):r for r in rows}
    assert len(keyed)==len(rows)
    groups=defaultdict(list)
    for row in rows:
        groups[row['model_id'],row['readout'],row['task_id']].append(row)
        if row['success']:
            x,y=row['source_properties'],row['representative_properties']
            assert all(y[p]<v if p=='mutagenicity' else y[p]>v for p,v in x.items())
            ri=mean(abs(y[p]-v)/abs(v) if v!=0 else (y[p]-.1)/.1 for p,v in x.items())
            close(ri,row['RI_official_code'])
            assert 0<=row['Sim_raw']<=1 and 1<=row['SAS_raw']<=10
            assert type(row['Novelty_official']) is bool
            assert row['Novelty_official']==(row['representative_canonical_smiles'] not in novelty_reference)
    metrics=json.loads((ROOT/'results/metrics.json').read_text())
    assert len(metrics['per_task'])==60
    for summary in metrics['per_task']:
        values=groups[summary['model_id'],summary['readout'],summary['task_id']]
        assert len(values)==COUNTS[summary['task_id']]==summary['n_inputs']
        good=[r for r in values if r['success']]
        assert len(good)==summary['n_success']
        close(summary['SR'],100*len(good)/len(values))
        close(summary['Validity_legacy'],100*mean(r['valid_legacy'] for r in values))
        close(summary['Validity_current'],100*mean(r['valid_current'] for r in values))
        for field in ['RI_official_code','Sim_raw','SAS_raw']:
            assert good
            close(summary[field],mean(r[field] for r in good))
        close(summary['Novelty_official'],100*mean(r['Novelty_official'] for r in good))
    for summary in metrics['macro']:
        tasks=TASKS if summary['scope']=='Macro10' else (TASKS[:5] if summary['scope']=='IND' else TASKS[5:])
        source=[r for r in metrics['per_task'] if r['model_id']==summary['model_id'] and r['readout']==summary['readout'] and r['task_id'] in tasks]
        assert len(source)==len(tasks)
        for field in ['SR','Validity_legacy','Validity_current','RI_official_code','Sim_raw','SAS_raw','Novelty_official']:
            close(summary[field],mean(r[field] for r in source))
    for summary in metrics['APS']:
        good=[r for r in groups[summary['model_id'],summary['readout'],summary['task_id']] if r['success']]
        close(summary['APS'],mean(r['representative_properties'][summary['property']] for r in good))
    output_counts={}
    for short in ['mistral','llama']:
        grouped=readlines(HERE/('grouped_inputs_'+short+'.jsonl.gz'))
        assert len(grouped)==3905
        for g in grouped:
            base=(g['model_id'],g['task_id'],g['input_uid'])
            for view in ['N20_R1_K1','N20_R20_K20','N400_R1_K1_reference']:
                row=keyed[base+(view,)]
                allowed=([g['winners'][0]] if view=='N20_R1_K1' else (g['winners'] if view=='N20_R20_K20' else [g['native_N400_reference']]))
                assert row['output_slots']==len(allowed)
                if row['success']:
                    match=next(w for w in allowed if w['candidate_id']==row['representative_candidate_id'])
                    assert row['representative_raw_smiles']==match['raw_smiles']
        selected=[p for p in pairs if p['model_id']=='gellmo_p6_'+short]
        assert len(selected)==3905
        for p in selected:
            for field,view in [('N20_K1_success','N20_R1_K1'),('N20_K20_success','N20_R20_K20'),('N400_K1_success','N400_R1_K1_reference')]:
                assert p[field]==keyed[p['model_id'],p['task_id'],p['input_uid'],view]['success']
            assert not p['N20_K1_success'] or p['N20_K20_success']
        output_counts[short]=dict(
            N20_K1_success=sum(p['N20_K1_success'] for p in selected),
            N20_K20_success=sum(p['N20_K20_success'] for p in selected),
            N400_K1_success=sum(p['N400_K1_success'] for p in selected),
            N20_K20_vs_N400_K1_gains=sum(p['N20_K20_success'] and not p['N400_K1_success'] for p in selected),
            N20_K20_vs_N400_K1_losses=sum(p['N400_K1_success'] and not p['N20_K20_success'] for p in selected),
            N20_K20_vs_N20_K1_gains=sum(p['N20_K20_success'] and not p['N20_K1_success'] for p in selected),
            N20_K20_vs_N20_K1_losses=sum(p['N20_K1_success'] and not p['N20_K20_success'] for p in selected))
    provenance=json.loads((ROOT/'results/provenance.json').read_text())
    for path,digest in provenance.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    report=dict(status='PASS_INDEPENDENT_FINAL_RESULTS_REVIEW',input_readouts=23430,paired_inputs=7810,task_groups=60,
                exact_representative_membership_verified=True,original_RI_all_successes_recomputed=True,
                all_task_macro_and_APS_values_recomputed=True,all_denominators_full=True,all_source_hashes_match=True,
                paired_counts=output_counts,
                interpretation='N20R20 versus N400R1 changes N and K together. Gains/losses are observed paired outcomes, not an assumed ordering or a causal isolated budget effect.',
                reviewed_evaluator_sha256=hashlib.sha256((ROOT/'evaluate_votes.py').read_bytes()).hexdigest())
    (HERE/'final_results_independent_review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
