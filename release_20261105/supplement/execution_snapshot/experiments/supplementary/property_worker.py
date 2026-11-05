"""Persistent local RPC invoking the original frozen CPU chemistry implementations."""
from common import *
import contextlib, subprocess, traceback
from bridges import got_property_module,ref_modules

def main():
 require_allocation();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 kind,task,directory=sys.argv[1:];directory=safe(Path(directory)/'placeholder').parent
 protocol=sys.stdout;service=None;models=[];counter=0
 from property_counts import Counts
 counts=Counts()
 if kind.endswith('eval'):counts.install_eval(REPO)
 else:counts.install_inf()
 if kind.startswith('got'):
  original=got_property_module()
  if kind=='got_eval':
   from producer_guard import install_got
   install_got(original)
   write(directory/'invalid_probe_guard.json',original.EvalService._supp_guard_receipt)
  service=(original.InfService if kind=='got_inf' else original.EvalService)(task,directory)
 elif kind.startswith('raw'):
  ref,bridge,svc=ref_modules()
  if kind=='raw_eval':
   from producer_guard import install
   install(svc.MODULES['standard_producer'])
   write(directory/'invalid_probe_guard.json',svc.MODULES['standard_producer']._supp_guard_receipt)
  if kind=='raw_inf':
   import admet_ai
   constructor=admet_ai.ADMETModel
   def persistent(**kwargs):
    signature=repr(sorted(kwargs.items()))
    if not models:models.append((signature,constructor(**kwargs)))
    assert models[0][0]==signature
    return models[0][1]
   admet_ai.ADMETModel=persistent
 else:raise ValueError(kind)
 for line in sys.stdin:
  q=json.loads(line)
  if q['op']=='stop':break
  started=time.perf_counter();counts.reset()
  try:
   with contextlib.redirect_stdout(sys.stderr):
    if kind.startswith('got'):value=service.handle(q)
    elif kind=='raw_inf':
     data=q['data'];d=directory/('batch_%06d'%counter);counter+=1
     request=d/'requests.json';write(request,data);old=sys.argv
     try:
      sys.argv=['admet_worker','--requests',str(request),'--output',str(d/'inf')];svc.MODULES['admet_worker'].main()
     finally:sys.argv=old
     value=read(d/'inf/admet_result.json');value['physical_persistent_model_loads']=len(models)
    else:
     data=q['data'];d=directory/('batch_%06d'%counter);counter+=1;d.mkdir(parents=True,exist_ok=False)
     svc.MODULES['standard_producer'].produce(data,q['admet'],d)
     responses=read(d/'responses_raw.json');filled=bridge.backfill(data['candidates'],data['requests'],data['mapping'],responses,data['cache_contract'])
     source={r['input_uid']:r for r in data['records']};outcomes=[]
     for c in filled:
      r=source[c['input_uid']];lab=labels(r,c)
      feedback=ref.feedback_from_filled(r,c,dict(SR_dir_success=lab['success'],SR_delta_success=lab['threshold_success']))
      outcomes.append(dict(filled_candidate=c,feedback=feedback,**lab))
     value=dict(outcomes=outcomes,responses=responses,identity=read(d/'standard_identity.json'))
   response=dict(ok=True,result=value,wall_seconds=time.perf_counter()-started,measured_evaluator_calls=counts.snapshot())
  except Exception as err:response=dict(ok=False,error=repr(err),traceback=traceback.format_exc(),wall_seconds=time.perf_counter()-started,measured_evaluator_calls=counts.snapshot())
  protocol.write(json.dumps(response,allow_nan=False)+'\n');protocol.flush()
  if not response['ok']:return
if __name__=='__main__':main()
