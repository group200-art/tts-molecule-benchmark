from common import *
from chemistry import Chemistry
require_allocation()
checks=[]
for b,method,mode,task in [('mistral','got','got','BDP'),('llama','got','got','HMPQ'),('mistral','reflexion','raw','BDP'),('llama','reflexion','raw','HMPQ')]:
 row=next(r for r in lines(OUT/'original'/(b+'__'+method+'.jsonl.gz')) if r['record']['task_id']==task);record=row['record'];original=row['candidates']
 cs=[dict(candidate_id=c['candidate_id'],input_uid=record['input_uid'],task_id=task,source_smiles=record['source_smiles'],original_slot_id=c['original_slot_id'],raw_smiles=c['raw_smiles'],raw_text=c.get('raw_text') or '') for c in original]
 directory=OUT/'property_checks'/('%s_%s_%s_%s'%(b,method,task,os.environ['SLURM_JOB_ID']));service=Chemistry(task,directory,mode)
 try:outcomes,cost=service.evaluate(record,cs,dict(arm='property_reproduction',access='offline',input_uid=record['input_uid']))
 finally:service.close()
 for old,new in zip(original,outcomes):
  filled=new['filled_candidate']
  for field in ['success','threshold_success','structural_valid']:assert old[field]==new[field],(b,method,field,old['candidate_id'])
  assert old['properties']==filled['properties'],(b,method,old['candidate_id'],old['properties'],filled['properties'])
 checks.append(dict(backbone=b,method=method,task=task,candidates=len(original),status='PASS'))
 log('property_reproduction_case',**checks[-1])
write(OUT/'property_checks.json',dict(status='PASS',cases=checks))
