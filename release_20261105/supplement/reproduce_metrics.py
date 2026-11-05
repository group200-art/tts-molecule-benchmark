"""Portable reproduction from complete input outcomes, without model or predictor calls."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

TASKS=['BDP','BDQ','BPQ','DPQ','BDPQ','MPQ','BDMQ','BHMQ','BMPQ','HMPQ']

def read_csv(path):
    with (gzip.open(path,'rt',newline='') if str(path).endswith('.gz') else open(path,newline='')) as f:
        return list(csv.DictReader(f))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--expected',required=True)
    p.add_argument('--paired',help='Optional expected paired contrasts to reproduce')
    p.add_argument('--manifest',help='CSV with the independently fixed task, instance_id and raw_source_id columns')
    p.add_argument('--expected-count',type=int,required=True);p.add_argument('--output',required=True)
    args=p.parse_args();rows=read_csv(args.data);reference=read_csv(args.expected)
    groups={}
    def group(r):return (r.get('analysis') or 'generation',r['backbone'],r.get('method') or r['arm'],r.get('selector') or '')
    for r in rows:
        k=group(r);key=(r['task'],r['instance_id']);g=groups.setdefault(k,{})
        if key in g:raise ValueError('Duplicate input: '+str((k,key)))
        g[key]=r
    first=next(iter(groups.values()));keys=sorted(first,key=lambda k:(TASKS.index(k[0]),k[1]))
    assert len(keys)==args.expected_count
    for g in groups.values():
        assert set(g)==set(keys),'Missing or extra inputs; no partial denominators permitted'
        assert all(g[k]['raw_source_id']==first[k]['raw_source_id'] for k in keys)
    if args.manifest:
        fixed=read_csv(args.manifest)
        fixed_keys={(r['task'],r['instance_id']):r['raw_source_id'] for r in fixed}
        assert len(fixed_keys)==len(fixed)==len(keys) and set(fixed_keys)==set(keys)
        assert all(fixed_keys[k]==first[k]['raw_source_id'] for k in keys)
    clusters=sorted({first[k]['raw_source_id'] for k in keys});lookup={k:i for i,k in enumerate(clusters)}
    ix=np.array([lookup[first[k]['raw_source_id']] for k in keys]);ti={t:np.array([i for i,k in enumerate(keys) if k[0]==t]) for t in TASKS}
    assert all(len(v)>0 for v in ti.values())
    rng=np.random.default_rng(20261014);accepted=[];n=0;rejected=0
    while n<2000:
        w=rng.multinomial(len(clusters),np.full(len(clusters),1/len(clusters)),size=2000-n)
        ok=np.ones(len(w),bool)
        for idx in ti.values():ok&=(w[:,ix[idx]].sum(axis=1)>0)
        rejected+=int((~ok).sum());accepted.append(w[ok]);n+=int(ok.sum())
    weights=np.concatenate(accepted);tw={t:weights[:,ix[idx]] for t,idx in ti.items()};den={t:w.sum(axis=1) for t,w in tw.items()}
    def estimate(values):
        y=np.asarray(values,float);assert y.shape==(len(keys),) and np.isfinite(y).all()
        point=sum(100*float(y[idx].mean()) for idx in ti.values())/10
        draws=sum(10*(tw[t]@y[idx])/den[t] for t,idx in ti.items())
        lo,hi=np.percentile(draws,[2.5,97.5]);return point,float(lo),float(hi)
    output=[];maximum_error=0.
    for r in reference:
        k=group(r);metric=r['metric'];g=groups[k]
        if r['estimate'] in ['NA','']:
            assert all(g[x].get(metric) in ['NA',''] for x in keys)
            output.append(dict(r));continue
        values=[float(g[x][metric]) for x in keys];result=estimate(values)
        for field,value in zip(['estimate','ci_low','ci_high'],result):
            error=abs(value-float(r[field]));maximum_error=max(maximum_error,error)
            assert error<1e-10,(k,metric,field,error)
        output.append(dict(r,**dict(zip(['estimate','ci_low','ci_high'],result))))
    paired=[]
    for r in read_csv(args.paired) if args.paired else []:
        analysis=r.get('analysis') or 'generation';b=r['backbone'];metric=r['metric']
        if analysis=='generation':
            assert r['contrast']=='GoT minus Sampling with GoT scoring'
            left=groups[analysis,b,'GoT',''];right=groups[analysis,b,'Sampling with GoT scoring','']
            values=[float(left[k][metric])-float(right[k][metric]) for k in keys]
        elif analysis=='similarity':
            assert r['contrast']=='Sim > 0.6 minus unconstrained'
            g=groups[analysis,b,r['method'],'']
            constrained=metric.replace('SR@','SR_sim>0.6@')
            values=[float(g[k][constrained])-float(g[k][metric]) for k in keys]
        else:
            left_name,right_name=r['contrast'].rsplit(' minus ',1)
            left=groups[analysis,b,r['method'],left_name];right=groups[analysis,b,r['method'],right_name]
            values=[float(left[k][metric])-float(right[k][metric]) for k in keys]
        result=estimate(values)
        for field,value in zip(['estimate','ci_low','ci_high'],result):
            error=abs(value-float(r[field]));maximum_error=max(maximum_error,error)
            assert error<1e-10,(r['contrast'],metric,field,error)
        paired.append(dict(r,**dict(zip(['estimate','ci_low','ci_high'],result))))
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    with (out/'recomputed.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
    if paired:
        with (out/'recomputed_paired.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
    report=dict(status='PASS',rows_checked=len(output),paired_contrasts_checked=len(paired),independent_input_manifest_checked=bool(args.manifest),input_count_per_configuration=len(keys),clusters=len(clusters),replicates=2000,bootstrap_seed=20261014,weight_sha256=hashlib.sha256(weights.astype('<i8').tobytes()).hexdigest(),zero_denominator_redraws=rejected,max_absolute_error=maximum_error,numpy_version=np.__version__,scope='Input-outcome aggregation and conditional source bootstrap; no new generation or chemical evaluation.')
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
