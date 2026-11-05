from common import *
from stats import Bootstrap
import numpy as np
require_allocation();records=subset();boot=Bootstrap(records,'stats_engineering_check')
y=np.array([(i%3)/2 for i in range(len(records))]);zero,rep=boot.estimate(np.zeros(len(records)));one,_=boot.estimate(np.ones(len(records)));diff=boot.difference(y,y)
assert zero['estimate']==zero['ci_low']==zero['ci_high']==0
assert one['estimate']==one['ci_low']==one['ci_high']==100
assert diff['estimate']==diff['ci_low']==diff['ci_high']==0
# Independent explicit recomputation of one globally sampled source replicate.
w=boot.weights[0];index={s:i for i,s in enumerate(boot.clusters)};manual=[]
for task in TASKS:
 rows=[(r,v) for r,v in zip(records,y) if r['task_id']==task];den=sum(w[index[r['source_molecule_uid']]] for r,v in rows);manual.append(100*sum(w[index[r['source_molecule_uid']]]*v for r,v in rows)/den)
_,rep=boot.estimate(y);assert abs(rep[0]-sum(manual)/10)<1e-12
# Identical source counts are shared across all seed contrasts, never flattened.
seeds=[y,-y,y*.5];est,_=boot.estimate(np.mean(seeds,axis=0));individual=[boot.estimate(s)[1] for s in seeds];assert np.allclose(np.mean(individual,axis=0),boot.estimate(np.mean(seeds,axis=0))[1])
write(OUT/'stats_checks.json',dict(status='PASS',source_cluster_weights_shared=True,task_denominators_recomputed=True,ten_task_macro=True,paired_zero_contrast=True,seed_average_inside_replicate=True,zero_denominator_rule='reject/redraw before aggregation',fixed_generation_inputs=len(records)))
log('statistics_checks_passed')
