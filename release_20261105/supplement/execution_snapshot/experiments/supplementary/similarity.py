"""Recompute full precision similarity in the unchanged evaluation environment."""
from common import *
from functools import lru_cache

def main():
 require_allocation()
 from rdkit import rdBase,Chem
 assert rdBase.rdkitVersion=='2020.03.3'
 sys.path.insert(0,str(REPO/'data/props'));import properties
 assert sha(properties.__file__)=='771722423bdb76955724e6d8a3c8298cedf29c55cf631b847640b4dfc6bec812'
 @lru_cache(maxsize=100000)
 def similarity(source,candidate):
  for raw in [source,candidate]:
   mol=Chem.MolFromSmiles(raw)
   if mol is None or mol.GetNumAtoms()==0:raise ValueError('Successful source/candidate cannot be parsed for full-precision similarity')
  return properties.similarity(source,candidate)
 for b in MODELS:
  for m in METHODS:
   errors=[];n=[0];candidate_rows=[];before=similarity.cache_info();started=time.perf_counter()
   def rows():
    for row in lines(OUT/'original'/('%s__%s.jsonl.gz'%(b,m))):
     record=row['record'];cs=row['candidates'];sims=[]
     for c in cs:
      sim=None
      if c['success']:
       try:
        sim=float(similarity(record['source_smiles'],c['raw_smiles']))
        if not math.isfinite(sim):raise ValueError('nonfinite similarity')
       except Exception as err:errors.append(dict(task_id=record['task_id'],input_uid=record['input_uid'],candidate_id=c['candidate_id'],error=repr(err)))
      sims.append(sim)
     if any(c['success'] and s is None for c,s in zip(cs,sims)):continue
     values={**{'SR@'+str(k):float(any(c['success'] for c in cs[:k])) for k in KS},'validity':sum(c['structural_valid'] for c in cs)/20.,'SR_delta@20':float(any(c['threshold_success'] for c in cs))}
     for tau in [.2,.4,.6,.8]:
      constrained=[c['success'] and s>tau for c,s in zip(cs,sims)]
      strong=[c['threshold_success'] and s>tau for c,s in zip(cs,sims)]
      for k in KS:
       val=float(any(constrained[:k]));assert val<=values['SR@'+str(k)];values['SR_sim>'+str(tau)+'@'+str(k)]=val
      values['SR_delta_sim>'+str(tau)+'@20']=float(any(strong))
     n[0]+=1
     yield dict(backbone=b,method=m,analysis='similarity',record=record,values=values,candidate_similarity=[dict(candidate_id=c['candidate_id'],original_position=i,success=c['success'],similarity_full_precision=s,reason=None if s is not None else 'not_needed_original_success_false') for i,(c,s) in enumerate(zip(cs,sims))])
   path=OUT/'offline_inputs'/('similarity__%s__%s.jsonl.gz'%(b,m));emit(path,rows())
   write(OUT/'offline_status'/('similarity__%s__%s.json'%(b,m)),dict(status='COMPLETE' if n[0]==3905 and not errors else 'BLOCKED',inputs=n[0],errors=errors,denominator_required=3905,output=str(path),similarity_cache=similarity.cache_info()._asdict(),rdkit=rdBase.rdkitVersion,actual_similarity_function_calls=similarity.cache_info().misses-before.misses,similarity_cache_hits=similarity.cache_info().hits-before.hits,wall_seconds=time.perf_counter()-started))
   log('similarity_completed',backbone=b,method=m,inputs=n[0],errors=len(errors))
if __name__=='__main__':main()
