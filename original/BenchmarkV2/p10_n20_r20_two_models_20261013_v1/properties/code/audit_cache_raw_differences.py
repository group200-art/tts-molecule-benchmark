"""Quantify every alternate cache's raw drift without changing selected values."""
import math
from collections import Counter,defaultdict
from common import *

def candidates(path):
    p=Path(path)
    if str(p).endswith('.jsonl.gz'):
        with gzip.open(str(p),'rt') as f:
            for line in f:
                r=json.loads(line);yield r['selected_candidate']
        return
    d=read(p)
    if isinstance(d,list):
        for r in d:
            if 'property_evidence'in r:yield r['property_evidence']
        return
    if 'by_budget'in d:
        groups=[g['inputs'] for g in d['by_budget'].values()]
    elif 'inputs'in d:groups=[d['inputs']]
    elif 'results'in d:
        for r in d['results']:yield dict(r,properties=r['comparison_properties'])
        return
    else:raise AssertionError(str(p))
    for group in groups:
        for r in group:
            for c in r['candidates']:yield c

def main():
    require_allocation();inventory=read(OUT/'reports/inventory.json');primary={}
    for task in TASKS:
        for r in read(OUT/'prepared'/(task+'.json.gz'))['cached_responses']:primary[r['request_id']]=r
    maxdiff=defaultdict(float);counts=Counter();unique=defaultdict(set);examples=defaultdict(list);comp_conflicts=[]
    for path,h in inventory['source_hashes'].items():
        assert sha(path)==h
        for c in candidates(path):
            rid=c.get('request_id')
            if rid not in primary:continue
            first=primary[rid];a,b=first['values'],c.get('raw_properties')
            if not isinstance(a,dict) or not isinstance(b,dict):continue
            if c.get('properties')!=first['cached_comparison_properties']:
                comp_conflicts.append(dict(request_id=rid,source=path));continue
            assert a.keys()==b.keys()
            for name,x in a.items():
                y=b[name]
                if x is None or y is None:assert x==y;continue
                assert math.isfinite(x) and math.isfinite(y)
                delta=abs(x-y)
                if delta:
                    counts[name+'_nonidentical_occurrences']+=1;unique[name].add(rid)
                    maxdiff[name]=max(maxdiff[name],delta)
                    if delta>1e-12:counts[name+'_above_1e12_occurrences']+=1
                    if len(examples[name])<5:examples[name].append(dict(request_id=rid,primary=x,alternate=y,absolute_difference=delta,alternate_source=path))
    result=dict(status='PASS' if not comp_conflicts and maxdiff['sim']<=1e-12 and maxdiff['sas']<=1e-12 else 'REVIEW_REQUIRED',comparison_conflicts=comp_conflicts,max_absolute_differences=dict(maxdiff),unique_request_counts={k:len(v) for k,v in unique.items()},occurrences=dict(counts),examples=dict(examples),raw_Sim_SAS_agree_within_1e12=maxdiff['sim']<=1e-12 and maxdiff['sas']<=1e-12,selection_policy='Predeclared source priority starts with original voting measurements; comparison equality is required. No success-based cache choice.',inference='Small nonidentical ADMET/standard prediction values may reflect historical floating-point serialization or numerical execution. This audit quantifies drift but does not claim to establish its cause.',inventory_sha256=sha(OUT/'reports/inventory.json'))
    write(OUT/'reports/raw_cache_difference_audit.json',result)
    assert result['status']=='PASS'
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
