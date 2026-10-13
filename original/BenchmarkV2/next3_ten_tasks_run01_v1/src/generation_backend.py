"""One standard P(6) load per logical run; direct measured PEFT generation."""
import time
from collections import Counter
from contracts import ROOT, read, write, digest
from runtime_common import require_allocation


def returned_record(input_ids, sequence, eos, cap):
    if sequence[:len(input_ids)]!=input_ids:
        raise ValueError('Sequence does not preserve input prefix')
    continuation=sequence[len(input_ids):]
    if not 1<=len(continuation)<=cap: raise ValueError('Returned token budget mismatch')
    counted=continuation.index(eos)+1 if eos in continuation else len(continuation)
    return {'sequence_ids':sequence,'continuation_ids':continuation,
            'tokens_through_first_eos':counted,'post_eos_tokens':len(continuation)-counted,
            'reached_cap_without_eos':counted==cap and eos not in continuation}


class RealGeneration:
    def __init__(self, method, cfg, costs, directory):
        self.method,self.cfg,self.costs,self.directory=method,cfg,costs,directory
        self.model=None

    def start(self):
        require_allocation()
        import torch, transformers, peft
        from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, set_seed
        from peft import PeftModel
        from dataclasses import asdict
        if self.model is not None: raise RuntimeError('Second model load forbidden')
        if torch.cuda.device_count()!=1 or not torch.cuda.is_bf16_supported():
            raise RuntimeError('Expected one allocated BF16 GPU')
        torch.cuda.set_device(0);torch.set_num_threads(8)
        gpu=torch.cuda.get_device_properties(0)
        if gpu.total_memory<40*1024**3: raise RuntimeError('GPU below authorized memory plan')
        set_seed(self.cfg['seed'])
        spec=read(ROOT/'configs/runtime.json')
        base_path=spec['base_path'];adapter_path=spec['adapter_path']
        fast=self.method!='p6_tot20'
        self.tok=AutoTokenizer.from_pretrained(base_path,use_fast=fast,local_files_only=True,trust_remote_code=False)
        if self.tok.is_fast!=fast or (self.tok.bos_token_id,self.tok.eos_token_id)!=(1,2):
            raise ValueError('Tokenizer differs from method contract')
        if self.tok.pad_token_id is None or fast: self.tok.pad_token_id=self.tok.eos_token_id
        if fast: self.tok.padding_side='left'
        elif self.tok.padding_side!='right': raise ValueError('ToT default padding differs')
        self.generation=GenerationConfig.from_pretrained(base_path,local_files_only=True)
        unused=self.generation.update(**dict(self.cfg['decoding'],bos_token_id=self.tok.bos_token_id,
                  eos_token_id=self.tok.eos_token_id,pad_token_id=self.tok.pad_token_id,return_dict_in_generate=False))
        if unused: raise ValueError('Unused decoding config fields')
        self.generation.validate()
        identity={'gpu':{'name':gpu.name,'total_memory_bytes':gpu.total_memory,'capability':[gpu.major,gpu.minor]},
                  'versions':{'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__},
                  'tokenizer':{'class':type(self.tok).__name__,'is_fast':self.tok.is_fast,
                   'padding_side':self.tok.padding_side,'bos_token_id':self.tok.bos_token_id,'eos_token_id':self.tok.eos_token_id,
                   'pad_token_id':self.tok.pad_token_id,'special_tokens':self.tok.special_tokens_map,
                   'add_bos_token':self.tok.add_bos_token,'add_eos_token':self.tok.add_eos_token,'size':len(self.tok),
                   'chat_template_applied':False},'generation_config_passed':self.generation.to_dict(),
                  'generate_other_kwargs':{'synced_gpus':False},'load_spec':spec,
                  'startup_seed':self.cfg['seed'],'seed_basis':'startup once, ToT additionally per node'}
        write(self.directory/'initialization.json',identity)
        torch.cuda.reset_peak_memory_stats(0);torch.cuda.synchronize(0);t=time.perf_counter()
        self.costs['load_started']+=1
        write(self.directory/'load_started.json',{'started':True,'wall_timestamp':time.time()})
        base=AutoModelForCausalLM.from_pretrained(base_path,torch_dtype=torch.bfloat16,
              device_map={'':'cuda:0'},local_files_only=True,trust_remote_code=False,use_safetensors=True)
        self.model=PeftModel.from_pretrained(base,adapter_path,adapter_name='default',is_trainable=False,
              autocast_adapter_dtype=True,low_cpu_mem_usage=False,ephemeral_gpu_offload=False,local_files_only=True)
        self.model.eval();torch.cuda.synchronize(0)
        self.costs['load_completed']+=1;self.costs['load_cuda_wall_seconds']=time.perf_counter()-t
        status=asdict(self.model.get_model_status())
        if status['enabled'] is not True or status['active_adapters']!=['default'] or status['merged_adapters']!=[]:
            raise RuntimeError('Loaded P(6) adapter is not active and unmerged')
        placements=Counter((str(p.device),str(p.dtype),'lora' if 'lora_' in n else 'base') for n,p in self.model.named_parameters())
        if any(dev!='cuda:0' for dev,_,_ in placements): raise RuntimeError('Unexpected model offload/device')
        identity.update(adapter_status=status,parameter_placements=[{'device':k[0],'dtype':k[1],'kind':k[2],'parameter_count':v} for k,v in placements.items()],
                        attention_implementation=self.model.config._attn_implementation,
                        attention_classes=dict(Counter(type(m).__name__ for m in self.model.modules() if 'Attention' in type(m).__name__)))
        write(self.directory/'model_identity.json',identity)

    def generate(self, prompt, context, seed=None):
        import torch
        from transformers import set_seed
        if seed is not None:
            if self.method!='p6_tot20': raise ValueError('Flat generation cannot reseed by input')
            set_seed(seed);torch.cuda.manual_seed_all(seed)
        ids=self.tok(prompt,add_special_tokens=True,truncation=False,return_attention_mask=True)
        input_ids=ids['input_ids']
        if not input_ids or any(v!=1 for v in ids['attention_mask']): raise ValueError('Invalid single prompt tokenization')
        if self.method=='p6_tot20' and len(input_ids)>2048: raise ValueError('ToT prompt exceeds2048; stop without truncation')
        if len(input_ids)+self.cfg['decoding']['max_new_tokens']>32768: raise ValueError('Context limit')
        callnum=len(self.costs['calls'])
        call={'call_index':callnum,'context':context,'seed':seed,'prompt':prompt,'prompt_digest':digest(prompt),
              'input_ids':input_ids,'attention_mask':ids['attention_mask'],'prompt_tokens_unexpanded':len(input_ids),
              'generation_config_passed':self.generation.to_dict(),'started':False,'completed':False,
              'exception':None,'cuda_wall_seconds':None}
        self.costs['calls'].append(call)
        # Legacy depth2 truncation is applied only after the complete length check.
        opts={'return_tensors':'pt','truncation':self.method=='p6_tot20' and context['stage']=='depth2'}
        if opts['truncation']: opts['max_length']=2048
        tokens=self.tok(prompt,**opts)
        if tokens['input_ids'][0].tolist()!=input_ids: raise ValueError('Prompt changed on tensor tokenization')
        tokens={k:v.to('cuda:0') for k,v in tokens.items()}
        torch.cuda.synchronize(0);t=time.perf_counter()
        self.costs['generate_started']+=1;call['started']=True
        write(self.directory/'calls'/('{:03d}_started.json'.format(callnum)),call)
        try:
            with torch.inference_mode():
                seq=self.model.generate(**tokens,generation_config=self.generation,synced_gpus=False)
            torch.cuda.synchronize(0);call['cuda_wall_seconds']=time.perf_counter()-t
            self.costs['generate_completed']+=1;call['completed']=True
            rows=seq.detach().cpu().tolist()
            self.costs['generated_slots_actual']+=len(rows)
            if len(rows)!=self.cfg['decoding']['num_return_sequences']: raise ValueError('Wrong physical return count')
            evidence=[]
            for row in rows:
                e=returned_record(input_ids,row,self.tok.eos_token_id,self.cfg['decoding']['max_new_tokens'])
                e['decoded_raw_with_special_tokens']=self.tok.decode(e['continuation_ids'],skip_special_tokens=False,clean_up_tokenization_spaces=False)
                e['decoded_continuation_untrimmed']=self.tok.decode(e['continuation_ids'],skip_special_tokens=True)
                evidence.append(e)
            call['outputs']=evidence
            self.costs['prompt_tokens_unexpanded']+=len(input_ids)
            self.costs['returned_tokens_through_first_eos']+=sum(e['tokens_through_first_eos'] for e in evidence)
            return [e['decoded_continuation_untrimmed'] for e in evidence]
        except Exception as error:
            call['exception']=repr(error);raise
        finally:
            if call['cuda_wall_seconds'] is None:
                torch.cuda.synchronize(0);call['cuda_wall_seconds']=time.perf_counter()-t
            self.costs['generate_cuda_wall_seconds']+=call['cuda_wall_seconds']
            write(self.directory/'calls'/('{:03d}_result.json'.format(callnum)),call)

    def memory(self):
        import torch
        self.costs['peak_allocated_bytes']=torch.cuda.max_memory_allocated(0)
        self.costs['peak_reserved_bytes']=torch.cuda.max_memory_reserved(0)
