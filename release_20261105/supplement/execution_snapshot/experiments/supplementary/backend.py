"""Frozen model assembly; batched sampling with a separate stable RNG per response."""
from common import *
import copy
from collections import Counter

class Model:
 def __init__(self,backbone,directory):
  require_allocation();self.backbone=backbone;self.directory=directory;directory.mkdir(parents=True,exist_ok=True);self.batch_index=0
  import torch,transformers,peft
  from transformers import AutoTokenizer,AutoModelForCausalLM,GenerationConfig
  from peft import PeftModel
  self.torch=torch
  assert (torch.__version__,transformers.__version__,peft.__version__)==('2.7.0+cu128','4.46.3','0.14.0')
  assert torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported()
  torch.set_num_threads(8);torch.cuda.set_device(0)
  gpu=torch.cuda.get_device_properties(0)
  assert gpu.name=='NVIDIA RTX PRO 6000 Blackwell Server Edition'
  spec=read((M if backbone=='mistral' else G)/'sources/runtime.json')
  self.tok=AutoTokenizer.from_pretrained(spec['base_path'],use_fast=True,local_files_only=True,trust_remote_code=False)
  self.tok.padding_side='left';self.tok.clean_up_tokenization_spaces=False
  if self.tok.pad_token_id is None:self.tok.pad_token_id=self.tok.eos_token_id
  base=AutoModelForCausalLM.from_pretrained(spec['base_path'],torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},local_files_only=True,trust_remote_code=False,use_safetensors=True)
  self.model=PeftModel.from_pretrained(base,spec['adapter_path'],adapter_name='default',is_trainable=False,autocast_adapter_dtype=True,low_cpu_mem_usage=False,ephemeral_gpu_offload=False,local_files_only=True);self.model.eval()
  self.cfg=GenerationConfig.from_dict(read(G/'configs/mistral_generation_defaults.json'))
  contract=read(G/'configs/design_contract.json')['decoding_contract']
  opts={k:contract[k] for k in ['do_sample','temperature','top_p','top_k','num_beams','num_beam_groups','num_return_sequences','repetition_penalty','no_repeat_ngram_size','length_penalty','min_new_tokens','min_length']}
  opts.update(bos_token_id=self.tok.bos_token_id,eos_token_id=self.tok.eos_token_id,pad_token_id=self.tok.pad_token_id,use_cache=True,return_dict_in_generate=False,forced_eos_token_id=None,forced_bos_token_id=None,stop_strings=None)
  assert not self.cfg.update(**opts)
  self.identity=dict(backbone=backbone,model=spec,gpu=dict(name=gpu.name,memory_bytes=gpu.total_memory),versions=dict(torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__),dtype='base BF16; adapters FP32',attention_implementation=self.model.config._attn_implementation,tokenizer=dict(class_name=type(self.tok).__name__,padding='left',bos=self.tok.bos_token_id,eos=self.tok.eos_token_id,pad=self.tok.pad_token_id,chat_template=False),batching=dict(max_responses=20,ordering='controller original trajectory/node order',random_streams='one CUDA torch.Generator per stable response hash; explicit multinomial dispatch'),generation_config=self.cfg.to_dict())
  identity_path=directory/'model_identity.json'
  if identity_path.exists():assert read(identity_path)==self.identity, 'Model/runtime identity changed during resume'
  else:write(identity_path,self.identity)
 def call(self,requests):
  """All requests in a physical batch have the same role cap; immutable per-response cache."""
  results=[]
  for start in range(0,len(requests),20):results.extend(self._batch(requests[start:start+20]))
  return results
 def _batch(self,requests):
  torch=self.torch;assert requests and len({r['max_new_tokens'] for r in requests})==1
  paths=[self.directory/'responses'/(digest(r)+'.json') for r in requests]
  found=[p.exists() for p in paths]
  if all(found):
   values=[read(p) for p in paths]
   assert all(v['request']==r for r,v in zip(requests,values));return values
  batch_id=digest(dict(requests=requests,model=self.identity));saved_batch=self.directory/'batches'/(batch_id+'.json')
  if any(found) or saved_batch.exists():
   saved=read(saved_batch)['responses']
   assert [v['request'] for v in saved]==requests
   for p,v in zip(paths,saved):
    if not p.exists():write(p,v)
   return saved
  tokens=self.tok([r['prompt'] for r in requests],add_special_tokens=True,padding=True,truncation=False,return_tensors='pt')
  width=tokens['input_ids'].shape[1];assert 0<width<=4096
  cap=requests[0]['max_new_tokens'];cfg=copy.deepcopy(self.cfg);cfg.max_new_tokens=cap;cfg.validate()
  gens=[torch.Generator(device='cuda').manual_seed(r['derived_seed']) for r in requests]
  before=[hashlib.sha256(bytes(g.get_state().tolist())).hexdigest() for g in gens]
  orig=torch.multinomial;draws=[0]*len(requests)
  def per_stream(prob,num_samples,replacement=False,*,generator=None,out=None):
   if prob.dim()!=2 or prob.shape[0]!=len(requests) or num_samples!=1 or out is not None or generator is not None:
    raise RuntimeError('Unexpected sampling operation: independent RNG contract would be violated')
   values=[]
   for i,g in enumerate(gens):values.append(orig(prob[i],num_samples,replacement=replacement,generator=g));draws[i]+=1
   return torch.stack(values)
  inputs={k:v.to('cuda:0') for k,v in tokens.items()};torch.cuda.synchronize();started=time.perf_counter()
  batch_id=digest(dict(requests=requests,model=self.identity))
  append(self.directory/'call_events.jsonl',dict(event='started',batch_id=batch_id,job_id=os.environ['SLURM_JOB_ID'],n_responses=len(requests),started_unix=time.time()))
  try:
   torch.multinomial=per_stream
   with torch.inference_mode():sequence=self.model.generate(**inputs,generation_config=cfg,synced_gpus=False)
   torch.cuda.synchronize()
  except BaseException as err:
   append(self.directory/'call_events.jsonl',dict(event='failed',batch_id=batch_id,error=repr(err),output_tokens=None))
   raise
  finally:torch.multinomial=orig
  elapsed=time.perf_counter()-started;rows=sequence.detach().cpu().tolist();batch_id=digest(dict(requests=requests,model=self.identity))
  values=[]
  for i,(r,ids) in enumerate(zip(requests,rows)):
   assert ids[:width]==tokens['input_ids'][i].tolist();cont=ids[width:];eos=self.tok.eos_token_id;n=cont.index(eos)+1 if eos in cont else len(cont);cont=cont[:n]
   raw=self.tok.decode(cont,skip_special_tokens=True,clean_up_tokenization_spaces=False)
   v=dict(record_id=digest(r),request=r,raw_text=raw,input_ids=tokens['input_ids'][i][tokens['attention_mask'][i].bool()].tolist(),output_ids=cont,input_tokens=int(tokens['attention_mask'][i].sum()),output_tokens=n,raw_with_special_tokens=self.tok.decode(cont,skip_special_tokens=False,clean_up_tokenization_spaces=False),rng_before=before[i],rng_after=hashlib.sha256(bytes(gens[i].get_state().tolist())).hexdigest(),random_draw_calls=draws[i],physical_batch_id=batch_id,physical_batch_size=len(requests),physical_batch_wall_seconds=elapsed,reached_cap_without_eos=n==cap and eos not in cont)
   values.append(v)
  # Commit complete physical batch first; partial response recovery can read this file.
  batch_path=self.directory/'batches'/(batch_id+'.json');write(batch_path,dict(responses=values,identity=self.identity,elapsed_seconds=elapsed))
  for p,v in zip(paths,values):write(p,v)
  append(self.directory/'call_events.jsonl',dict(event='completed',batch_id=batch_id,job_id=os.environ['SLURM_JOB_ID'],output_tokens=sum(v['output_tokens'] for v in values)))
  self.batch_index+=1;append(self.directory/'batches.jsonl',dict(batch_id=batch_id,n_responses=len(values),seconds=elapsed,input_tokens=sum(v['input_tokens'] for v in values),output_tokens=sum(v['output_tokens'] for v in values),batch_path=str(batch_path)))
  return values
