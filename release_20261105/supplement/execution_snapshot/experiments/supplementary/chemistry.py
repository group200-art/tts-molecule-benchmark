"""Versioned chemistry, separated inference/offline access and measured request logs."""
from common import *
import subprocess
class RPC:
 def __init__(self,kind,task,directory):
  directory.mkdir(parents=True,exist_ok=True);self.directory=directory;self.kind=kind
  self.stderr=(directory/'stderr.log').open('ab');py=INF_PY if kind.endswith('inf') else EVAL_PY
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
  self.proc=subprocess.Popen([py,'-B',str(CODE/'property_worker.py'),kind,task,str(directory)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,universal_newlines=True,env=env)
 def request(self,data):
  self.proc.stdin.write(json.dumps(data,allow_nan=False)+'\n');self.proc.stdin.flush();line=self.proc.stdout.readline()
  if not line:raise RuntimeError('Chemistry RPC exited; see '+str(self.directory))
  r=json.loads(line);self.last_counts=r.get('measured_evaluator_calls',{});append(self.directory/'rpc.jsonl',dict(operation=data['op'],ok=r['ok'],wall_seconds=r['wall_seconds'],measured_evaluator_calls=self.last_counts))
  if not r['ok']:raise RuntimeError(r)
  return r['result'],r['wall_seconds']
 def close(self):
  if self.proc.poll() is None:
   self.proc.stdin.write('{"op":"stop"}\n');self.proc.stdin.flush();self.proc.stdin.close()
   try:self.proc.wait(timeout=20)
   except subprocess.TimeoutExpired:self.proc.terminate();self.proc.wait(timeout=10)
  self.stderr.close()

class Chemistry:
 def __init__(self,task,directory,mode):
  self.task=task;self.directory=directory;self.mode=mode;self.services={};self.cache={};self.call_index=0
  for part in ('inf','eval'):self.services[part]=RPC(('got' if mode=='got' else 'raw')+'_'+part,task,directory/part)
  if mode=='got':
   self.identity={part:self.services[part].request(dict(op='identity'))[0] for part in ('inf','eval')}
  else:self.identity=dict(protocol='original raw-SMILES Reflexion/Vote property pipeline',code=read(OUT/'version_and_hashes.json')['original_code'])
  write(directory/'identity.json',self.identity)
 def evaluate(self,record,candidates,context):
  # Scoped arm input caches: shared initial data are copied explicitly by the controller.
  assert len(candidates)<=100
  if self.mode=='got':return self.got(record,candidates,context)
  return self.raw(record,candidates,context)
 def raw(self,record,candidates,context):
  from bridges import ref_modules
  ref,bridge,svc=ref_modules()
  spec=read(B/'p6_llama_main_ten_tasks_v1/configs/properties_reflexion.json')['tasks'][self.task]
  requests,mapping=bridge.requests_for(candidates,spec['cache_contract'])
  data=dict(method='reflexion',task_id=self.task,kind='feedback',records=[record],candidates=candidates,requests=requests,mapping=mapping,**spec)
  # Original producers deduplicate requests within each batch. No hidden cross-arm cache.
  admet,ti=self.services['inf'].request(dict(op='evaluate',data=data))
  result,te=self.services['eval'].request(dict(op='evaluate',data=data,admet=admet))
  n=sum(r['values'] is not None for r in result['responses']);ap=admet['identity']['unique_predictable_strings'];adcols=spec['admet_columns']
  scalar={p:(ap if p in adcols else n) for p in list(record['source_properties'])+['sim','sas']}
  cost=dict(candidate_evaluation_attempts=len(candidates),property_bundle_requests=len(candidates),unique_bundle_requests=len(requests),cache_hits=len(candidates)-len(requests),actual_scalar_computations=scalar,actual_batch_calls={p:int(ap>0) if p in adcols else n for p in scalar},source_property_gets=1,similarity_computations=n,SAS_computations=n,wall_seconds=ti+te,mode=context.get('access','inference'),identity=self.identity)
  cost['measured_evaluator_calls']={part:self.services[part].last_counts for part in ['inf','eval']}
  cost['SAS_computations']=cost['measured_evaluator_calls']['eval'].get('SAS_primitive_including_plogp',{}).get('completed',0)
  self.record_cost(context,cost)
  return result['outcomes'],cost
 def got(self,record,candidates,context):
  checked,ti=self.services['inf'].request(dict(op='recheck',nodes=[dict(node_id=c['candidate_id'],parsed_smiles=c['raw_smiles']) for c in candidates]))
  contract=read(G/'sources/properties.json')['tasks'][self.task]['cache_contract']
  contract=dict(contract,identity=self.identity,scorer_sha256=sha(G/'code/scorer.py'),property_service_sha256=sha(G/'code/property_service.py'))
  scope=(record['input_uid'],context['arm'])
  if scope not in self.cache:self.cache[scope]=dict(self.cache.get((record['input_uid'],'common_initial'),{}))
  cache=self.cache[scope]
  reqs={};mapping=[]
  for r in checked:
   identity=dict(task_id=self.task,source_smiles=record['source_smiles'],request_smiles=r['request_smiles'],cache_contract=contract)
   rid='supp-got-property-'+digest(identity);mapping.append((rid,r))
   if rid not in reqs and rid not in cache:reqs[rid]=dict(request_id=rid,identity=identity,rdkit_valid=r['rdkit_valid'],inf_canonical=r['inf_canonical'],property_error=r.get('property_error'))
  requests=list(reqs.values());strings=list(dict.fromkeys(r['identity']['request_smiles'] for r in requests if r['rdkit_valid'] and not r.get('property_error')))
  admet,tp=self.services['inf'].request(dict(op='predict',strings=strings,context=context))
  for r in requests:r['admet_raw_targets']=admet['by_smiles'].get(r['identity']['request_smiles'],{})
  results,te=self.services['eval'].request(dict(op='evaluate',requests=requests,context=context));cache.update({r['request_id']:r for r in results});byid=cache
  out=[]
  for c,(rid,recheck) in zip(candidates,mapping):
   filled=dict(c,**{k:v for k,v in byid[rid].items() if k not in c});filled.update(node_id=c['candidate_id'],parsed_smiles=c['raw_smiles'],raw_smiles=c['raw_smiles'],inf_rdkit_version='2024.03.6',canonical_representations=[dict(environment='gellmo_inf',valid=recheck['rdkit_valid'],smiles=recheck['inf_canonical'],rdkit_version='2024.03.6'),dict(environment='gellmo_eval_py36',valid=byid[rid]['eval_valid'],smiles=byid[rid]['eval_canonical'],rdkit_version='2020.03.3')]);lab=labels(record,filled)
   out.append(dict(filled_candidate=filled,**lab))
  n=sum(r['raw_properties'] is not None for r in results);adcols=read(G/'sources/properties.json')['tasks'][self.task]['admet_columns']
  cost=dict(candidate_evaluation_attempts=len(candidates),property_bundle_requests=len(candidates),unique_bundle_requests=len(requests),cache_hits=len(candidates)-len(requests),actual_scalar_computations={p:len(strings) if p in adcols else n for p in list(record['source_properties'])+['sim','sas']},actual_batch_calls={p:int(bool(strings)) if p in adcols else n for p in list(record['source_properties'])+['sim','sas']},source_property_gets=1,similarity_computations=n,SAS_computations=n,wall_seconds=ti+tp+te,mode=context.get('access','inference'),identity=self.identity)
  cost['measured_evaluator_calls']={part:self.services[part].last_counts for part in ['inf','eval']}
  cost['SAS_computations']=cost['measured_evaluator_calls']['eval'].get('SAS_primitive_including_plogp',{}).get('completed',0)
  self.record_cost(context,cost);return out,cost
 def record_cost(self,context,cost):
  append(self.directory/'workload.jsonl',dict(context=context,cost=cost,index=self.call_index));self.call_index+=1
 def close(self):
  for s in self.services.values():s.close()
