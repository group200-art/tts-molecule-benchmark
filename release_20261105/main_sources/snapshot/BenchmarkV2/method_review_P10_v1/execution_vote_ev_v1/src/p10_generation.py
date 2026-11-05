"""Fixed P6/20-return generation. No chemistry, vote, property, or success imports."""
import gzip
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from contracts import (ROOT,RUN_ID,read,write,sha,digest,seed_for,candidate_identity,require_allocation,
                       verify_code,isolated_env,append_event,destination)
from p10_parser import extract_candidate


def write_gzip(path,value):
    p=destination(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:
        with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:
            z.write(json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',', ':')).encode())


def token_record(sequence,prompt_ids,eos=2,cap=256):
    if sequence[:len(prompt_ids)]!=prompt_ids:raise ValueError('Returned prompt prefix changed')
    ids=sequence[len(prompt_ids):]
    if not 1<=len(ids)<=cap:raise ValueError('Returned continuation length outside cap')
    count=ids.index(eos)+1 if eos in ids else len(ids)
    if any(x!=eos for x in ids[count:]):raise ValueError('Non-padding after first EOS')
    return {'continuation_ids':ids,'tokens_through_first_eos':count,'post_eos_padding_tokens':len(ids)-count,
            'reached_cap_without_eos':count==cap and eos not in ids}


def set_all_seeds(seed):
    import numpy as np
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.cuda.manual_seed(seed);torch.cuda.manual_seed_all(seed)


class Generation:
    def __init__(self,directory,costs):self.d=directory;self.costs=costs

    def start(self):
        import torch,transformers,peft,numpy as np
        from transformers import AutoTokenizer,AutoModelForCausalLM,GenerationConfig
        from peft import PeftModel
        from dataclasses import asdict
        versions={'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,
                  'peft':peft.__version__,'numpy':np.__version__}
        write(self.d/'software_identity.json',versions)
        if (versions['python'],versions['torch'],versions['transformers'],versions['peft'])!=('3.10.20','2.7.0+cu128','4.46.3','0.14.0'):
            raise RuntimeError('Fixed generation software identity mismatch')
        if torch.cuda.device_count()!=1 or not torch.cuda.is_bf16_supported():raise RuntimeError('Expected one allocated BF16 GPU')
        torch.cuda.set_device(0);torch.set_num_threads(8)
        gpu=torch.cuda.get_device_properties(0)
        device={'name':gpu.name,'total_memory_bytes':gpu.total_memory,'capability':[gpu.major,gpu.minor]}
        write(self.d/'device_identity.json',device)
        if gpu.name!='NVIDIA H100 80GB HBM3' or not 75*1024**3<gpu.total_memory<82*1024**3:raise RuntimeError('Required H100 80GB HBM3 not allocated')
        p=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,driver_version','--format=csv,noheader'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
        write(self.d/'nvidia_smi_identity.json',{'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,
                                              'visible_device':os.environ.get('CUDA_VISIBLE_DEVICES'),'basis':'Inventory only; torch must see exactly one allocated device'})
        spec=read(ROOT/'configs/runtime.json');cfg=read(ROOT/'configs/generation.json')
        self.tok=AutoTokenizer.from_pretrained(spec['base_path'],use_fast=True,local_files_only=True,trust_remote_code=False)
        self.tok.pad_token_id=self.tok.eos_token_id;self.tok.padding_side='left'
        tokid={'class':type(self.tok).__name__,'is_fast':self.tok.is_fast,'padding_side':self.tok.padding_side,
               'bos_token_id':self.tok.bos_token_id,'eos_token_id':self.tok.eos_token_id,'pad_token_id':self.tok.pad_token_id,
               'add_bos_token':self.tok.add_bos_token,'add_eos_token':self.tok.add_eos_token,'size':len(self.tok),'chat_template_applied':False}
        if tokid!={'class':'LlamaTokenizerFast','is_fast':True,'padding_side':'left','bos_token_id':1,'eos_token_id':2,
                   'pad_token_id':2,'add_bos_token':True,'add_eos_token':False,'size':32768,'chat_template_applied':False}:
            raise RuntimeError('Tokenizer contract mismatch')
        self.generation=GenerationConfig.from_dict(cfg);self.generation.validate()
        actual=self.generation.to_dict()
        if actual!=cfg:raise RuntimeError('Complete effective generation configuration differs')
        self.identity={'gpu':device,'versions':versions,'tokenizer':tokid,'load_spec':spec,'generation_config_passed':actual,
                       'generate_other_kwargs':{'synced_gpus':False},'generation_config_sha256':sha(ROOT/'configs/generation.json')}
        write(self.d/'initialization.json',self.identity)
        self.costs['model_load_started']=1;write(self.d/'model_load_started.json',{'wall_timestamp':time.time()})
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter()
        base=AutoModelForCausalLM.from_pretrained(spec['base_path'],torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},
                     local_files_only=True,trust_remote_code=False,use_safetensors=True)
        self.model=PeftModel.from_pretrained(base,spec['adapter_path'],adapter_name='default',is_trainable=False,
                   autocast_adapter_dtype=True,low_cpu_mem_usage=False,ephemeral_gpu_offload=False,local_files_only=True)
        self.model.eval();torch.cuda.synchronize()
        self.costs['model_load_completed']=1;self.costs['model_load_wall_seconds']=time.perf_counter()-t
        status=asdict(self.model.get_model_status())
        if status['enabled'] is not True or status['active_adapters']!=['default'] or status['merged_adapters']!=[]:raise RuntimeError('Adapter must remain active and unmerged')
        places=Counter((str(p.device),str(p.dtype),'lora' if 'lora_' in n else 'base') for n,p in self.model.named_parameters())
        if any(dev!='cuda:0' or (kind=='base' and dtype!='torch.bfloat16') or (kind=='lora' and dtype!='torch.float32') for dev,dtype,kind in places):
            raise RuntimeError('Wrong model device/dtype or offload')
        self.identity.update(adapter_status=status,parameter_placements=[{'device':k[0],'dtype':k[1],'kind':k[2],'parameter_count':v} for k,v in places.items()],
                             attention_implementation=self.model.config._attn_implementation,training=self.model.training)
        write(self.d/'model_identity.json',self.identity)

    def one_input(self,record):
        import torch
        folder=self.d/'inputs'/record['task_id']/('{:04d}'.format(record['input_idx']));folder.mkdir(parents=True,exist_ok=False)
        tokens=self.tok(record['prompt'],add_special_tokens=True,truncation=False,return_attention_mask=True,return_tensors='pt')
        ids=tokens['input_ids'][0].tolist();mask=tokens['attention_mask'][0].tolist()
        if not ids or len(ids)+256>32768 or any(x!=1 for x in mask):raise RuntimeError('Prompt context/attention contract violation')
        if hashlib.sha256(record['prompt'].encode()).hexdigest()!=record['prompt_sha256']:raise ValueError('Prompt SHA changed')
        write(folder/'prompt_tokens.json',{'input_uid':record['input_uid'],'prompt_sha256':record['prompt_sha256'],
              'prompt_reference':str(ROOT/'configs/prompts.json'),'input_ids':ids,'attention_mask':mask,'add_special_tokens':True,'truncation':False})
        tokens={k:v.to('cuda:0') for k,v in tokens.items()};calls=[]
        for b in range(20):
            seed=seed_for(record['task_ord'],record['input_idx'],b)
            call={'task_id':record['task_id'],'task_ord':record['task_ord'],'input_idx':record['input_idx'],'input_uid':record['input_uid'],
                  'block':b,'seed':seed,'call_id':record['input_uid']+':block{:02d}'.format(b),'prompt_sha256':record['prompt_sha256'],
                  'prompt_tokens_unexpanded':len(ids),'prompt_tokens_expanded':20*len(ids),'physical_return_sequences':20,
                  'config_sha256':self.identity['generation_config_sha256'],'started':True,'completed':False,
                  'wall_timestamp':time.time(),'cuda_wall_seconds':None,'exception':None}
            self.costs['generate_started']+=1;append_event(self.d/'calls.jsonl',dict(call,event='started'))
            # Reset each RNG immediately before the one authorized physical call.
            set_all_seeds(seed);torch.cuda.synchronize();t=time.perf_counter()
            try:
                with torch.inference_mode():result=self.model.generate(**tokens,generation_config=self.generation,synced_gpus=False)
                torch.cuda.synchronize();elapsed=time.perf_counter()-t
                sequences=result.sequences.detach().cpu().tolist()
                # Preserve every returned token before validation/decoding can fail.
                rawpath=folder/('block{:02d}.returned.json.gz'.format(b))
                write_gzip(rawpath,{'call_id':call['call_id'],'prompt_tokens_path':str(folder/'prompt_tokens.json'),
                     'sequences_include_prompt':False,'continuation_ids':[row[len(ids):] for row in sequences],
                     'unexpected_full_sequences':{str(s):row for s,row in enumerate(sequences) if row[:len(ids)]!=ids},
                     'prompt_prefix_matches':[row[:len(ids)]==ids for row in sequences]})
                if len(sequences)!=20:raise ValueError('Physical return count differs from20')
                rows=[]
                for s,sequence in enumerate(sequences):
                    evidence=token_record(sequence,ids);n=evidence['tokens_through_first_eos']
                    untrimmed=self.tok.decode(evidence['continuation_ids'][:n],skip_special_tokens=True,clean_up_tokenization_spaces=False)
                    raw=untrimmed.strip();parsed=extract_candidate(raw);slot=20*b+s
                    evidence.pop('continuation_ids')
                    row=dict(candidate_identity(record,slot),global_slot=slot,block=b,block_slot=s,seed=seed,input_idx=record['input_idx'],
                             task_ord=record['task_ord'],raw_decode_untrimmed=untrimmed,raw_response=raw,
                             raw_response_sha256=hashlib.sha256(raw.encode()).hexdigest(),prompt_sha256=record['prompt_sha256'],
                             tokens_path=str(rawpath),tokens_row=s,**evidence,**parsed)
                    rows.append(row)
                proposalpath=folder/('block{:02d}.proposals.json.gz'.format(b));write_gzip(proposalpath,rows)
                call.update(completed=True,cuda_wall_seconds=elapsed,tokens_through_first_eos=sum(r['tokens_through_first_eos'] for r in rows),
                            post_eos_padding_tokens=sum(r['post_eos_padding_tokens'] for r in rows),
                            proposals_path=str(proposalpath),tokens_path=str(rawpath),proposals_sha256=sha(proposalpath),tokens_sha256=sha(rawpath))
                self.costs['generate_completed']+=1;self.costs['proposals']+=len(rows);self.costs['generated_tokens']+=call['tokens_through_first_eos']
                self.costs['generate_cuda_wall_seconds']+=elapsed
            except BaseException as error:
                call['exception']=repr(error)
                if call['cuda_wall_seconds'] is None:call['cuda_wall_seconds']=time.perf_counter()-t
                raise
            finally:append_event(self.d/'calls.jsonl',dict(call,event='result'))
            calls.append(call)
        write(folder/'complete.json',{'input_uid':record['input_uid'],'calls':calls,'proposals':400})
        self.costs['inputs_complete']+=1
        print(json.dumps({'event':'input_complete','task':record['task_id'],'index':record['input_idx'],'inputs_complete':self.costs['inputs_complete']}),flush=True)


def main():
    unit=require_allocation('gpu');verify_code()
    d=destination(ROOT/'runs'/unit['unit_id']);d.mkdir(parents=True,exist_ok=False)
    os.environ.update(isolated_env(d));os.chdir(str(d))
    costs={'model_load_started':0,'model_load_completed':0,'model_load_wall_seconds':None,'generate_started':0,'generate_completed':0,
           'proposals':0,'generated_tokens':0,'generate_cuda_wall_seconds':0.,'inputs_complete':0,'planned_calls':unit['planned_calls']}
    state={'unit':unit,'run_id':RUN_ID,'status':'started','code_inventory_sha256':sha(ROOT/'evidence/code_inventory_before_jobs.json'),
           'slurm':{k:v for k,v in os.environ.items() if k.startswith('SLURM_') or k=='CUDA_VISIBLE_DEVICES'}}
    write(d/'started.json',state);start=time.perf_counter()
    try:
        generator=Generation(d,costs);generator.start()
        prompts={r['input_uid']:r for r in read(ROOT/'configs/prompts.json')}
        for ref in unit['inputs']:
            r=prompts[ref['input_uid']]
            if any(r[k]!=v for k,v in ref.items()):raise ValueError('Unit/prompt mapping changed')
            generator.one_input(r)
        if costs['generate_completed']!=unit['planned_calls'] or costs['proposals']!=unit['planned_proposals']:raise ValueError('Incomplete generation unit')
        state['status']='pass'
    except BaseException as error:
        state['status']='failed';state['exception']=repr(error)
        write(d/'exception.json',{'exception':repr(error),'traceback':traceback.format_exc()});raise
    finally:
        costs['worker_wall_seconds']=time.perf_counter()-start
        if 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized():
            costs['peak_allocated_bytes']=sys.modules['torch'].cuda.max_memory_allocated()
            costs['peak_reserved_bytes']=sys.modules['torch'].cuda.max_memory_reserved()
        write(d/'costs.json',costs);write(d/'status.json',state)


if __name__=='__main__':main()
