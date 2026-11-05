"""Paired global raw-source bootstrap. Scores may be binary or analytic expectations."""
from common import *
import numpy as np

def permutation_success(n,s,k):
 assert 0<=s<=n and 0<k<=n
 return 1.0 if k>n-s else 1-math.comb(n-s,k)/math.comb(n,k)
def vote_uniform(counts,k,size=20):
 return 1-math.prod(1-s/size for s in counts[:k])
def constrained(success,sim,tau):
 if not success:return False
 if not isinstance(sim,(int,float)) or isinstance(sim,bool) or not math.isfinite(sim):raise ValueError('Successful candidate missing finite similarity')
 return sim>tau

class Bootstrap:
 def __init__(self,records,label,extra_denominators=None):
  self.records=records;self.label=label;self.clusters=sorted({r['source_molecule_uid'] for r in records});lookup={v:i for i,v in enumerate(self.clusters)}
  self.index=np.array([lookup[r['source_molecule_uid']] for r in records]);self.tasks={t:np.array([i for i,r in enumerate(records) if r['task_id']==t]) for t in TASKS}
  assert all(len(v) for v in self.tasks.values())
  rng=np.random.default_rng(20261014);accepted=[];rejected=0
  while sum(len(x) for x in accepted)<2000:
   need=2000-sum(len(x) for x in accepted)
   w=rng.multinomial(len(lookup),np.full(len(lookup),1/len(lookup)),size=need)
   good=np.ones(need,dtype=bool)
   for t,idx in self.tasks.items():good&=w[:,self.index[idx]].sum(axis=1)>0
   if extra_denominators is not None:
    for idx,den in extra_denominators:
     good&=(w[:,self.index[idx]]@den)>0
   rejected+=int((~good).sum());accepted.append(w[good])
  self.weights=np.concatenate(accepted,axis=0);self.taskweights={t:self.weights[:,self.index[idx]] for t,idx in self.tasks.items()};self.denoms={t:w.sum(axis=1) for t,w in self.taskweights.items()}
  self.metadata=dict(seed=20261014,replicates=2000,clusters=len(lookup),inputs=len(records),weight_sha256=hashlib.sha256(self.weights.astype('<i8').tobytes()).hexdigest(),zero_denominator_redraws=rejected,raw_source_unit='original/raw SMILES identifiers',tasks=TASKS,task_weighting='equal',source_weights_shared_across_methods_tasks_arms_seeds=True,conditional_on_saved_generation_runs=True,pointwise=True,multiple_comparison_adjustment=False)
  write(OUT/'bootstrap'/(label+'.json'),self.metadata)
 def estimate(self,scores):
  y=np.asarray(scores,dtype=float);assert y.shape==(len(self.records),) and np.isfinite(y).all()
  means=[];replicates=np.zeros(2000)
  for t,idx in self.tasks.items():
   means.append(100*float(y[idx].mean()));replicates+=10*(self.taskweights[t]@y[idx])/self.denoms[t]
  lo,hi=np.percentile(replicates,[2.5,97.5]);return dict(estimate=sum(means)/10,ci_low=float(lo),ci_high=float(hi)),replicates
 def difference(self,a,b):return self.estimate(np.asarray(a)-np.asarray(b))[0]
