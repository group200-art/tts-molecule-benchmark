"""Measure evaluator calls without modifying arguments, outputs or precision."""
from collections import Counter
import copy,sys

class Counts:
 def __init__(self):self.values={};self.versions={}
 def wrap(self,obj,name,label):
  original=getattr(obj,name)
  def measured(*args,**kwargs):
   row=self.values.setdefault(label,dict(started=0,completed=0,failed=0,scalar_outputs=0,batch_sizes=[]));row['started']+=1
   try:
    result=original(*args,**kwargs);row['completed']+=1
    if label=='ADMET_predict':
     row['batch_sizes'].append(len(result));row['scalar_outputs']+=int(result.size);row['output_columns']=list(result.columns)
    else:row['scalar_outputs']+=1
    return result
   except BaseException:
    row['failed']+=1;raise
  setattr(obj,name,measured)
 def install_eval(self,repo):
  sys.path.insert(0,str(repo/'data/props'));import properties
  for name,label in [('similarity','similarity'),('sas','sas_descriptor'),('drd2','drd2'),('qed','qed'),('penalized_logp','plogp')]:self.wrap(properties,name,label)
  self.wrap(properties.sascorer,'calculateScore','SAS_primitive_including_plogp')
 def install_inf(self):
  from admet_ai import ADMETModel
  self.wrap(ADMETModel,'predict','ADMET_predict')
 def reset(self):self.values={}
 def snapshot(self):return copy.deepcopy(self.values)
