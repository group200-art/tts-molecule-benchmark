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
        allowed=('RTX PRO 6000 Blackwell Server Edition' in gpu.name or
                 (('A100' in gpu.name or 'H100' in gpu.name) and gpu.total_memory>=78*1024**3))
        if not allowed: raise RuntimeError('Only approved Blackwell, A100 80GB or Sycamore H100 80GB hardware allowed')
        set_seed(51001)
        spec=read(ROOT/'configs/runtime.json')
        base_path=spec['base_path'];adapter_path=spec['adapter_path']
        fast=self.cfg['tokenizer_fast']
        self.tok=AutoTokenizer.from_pretrained(base_path,use_fast=fast,local_files_only=True,trust_remote_code=False)
        if self.tok.is_fast!=fast or (self.tok.bos_token_id,self.tok.eos_token_id)!=(1,2):
            raise ValueError('Tokenizer differs from method contract')
        if self.tok.pad_token_id is None or fast: self.tok.pad_token_id=self.tok.eos_token_id
        if fast: self.tok.padding_side='left'
        elif self.tok.padding_side!='right': raise ValueError('ToT default padding differs')
        self.generation=GenerationConfig.from_pretrained(base_path,local_files_only=True)
        self.base_generation_dict=self.generation.to_dict()
        unused=self.generation.update(**dict(self.cfg['decoding'],max_new_tokens=250,bos_token_id=self.tok.bos_token_id,
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
                  'startup_seed':51001,'seed_basis':'per-call approved SHA seed resets Python/NumPy/Torch/CUDA; startup seed has no call-stream role'}
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

    def generate(self, role, prompt, context, seed):
        import random
        import numpy as np
        import torch
        from transformers import GenerationConfig
        from self_refine import call_seed
        cap = self.cfg['max_new_tokens'][role]
        expected = call_seed(context['rep_seed'],context['task_id'],context['input_uid'],role,context['r'])
        if seed != expected or context['role'] != role:
            raise ValueError('Call seed/role differs from contract')
        if self.costs['generate_started'] >= self.costs['maximum_generate_calls']:
            raise RuntimeError('Physical call budget exceeded')
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        input_ids = self.tok(prompt,add_special_tokens=True,truncation=False)['input_ids']
        if not input_ids or len(input_ids)+cap > 32768: raise ValueError('Context bounds; no truncation')
        config = GenerationConfig.from_dict(self.base_generation_dict)
        spec = dict(self.cfg['decoding'],max_new_tokens=cap,bos_token_id=1,eos_token_id=2,
                    pad_token_id=2,return_dict_in_generate=False)
        if config.update(**spec): raise ValueError('Unused generation fields')
        config.validate()
        number = self.costs['generate_started']
        stem = '{:04d}'.format(number)
        detail = dict(call_index=number,context=context,seed=seed,prompt=prompt,prompt_digest=digest(prompt),
                      input_ids=input_ids,attention_mask=[1]*len(input_ids),
                      input_tokens=len(input_ids),untruncated=True,generation_config_passed=config.to_dict(),
                      started_at_unix=time.time(),completed=False)
        start_path = self.directory/'calls'/(stem+'_started.json')
        write(start_path,detail,compact=True)
        self.costs['generate_started'] += 1
        self.costs['prompt_tokens_unexpanded'] += len(input_ids)
        tokens = torch.tensor([input_ids],dtype=torch.long,device='cuda:0')
        torch.cuda.synchronize(0); started = time.perf_counter()
        result = dict(call_index=number,started_record=str(start_path),completed=False,exception=None)
        try:
            with torch.inference_mode():
                seq = self.model.generate(input_ids=tokens,attention_mask=torch.ones_like(tokens),
                                          generation_config=config,synced_gpus=False)
            torch.cuda.synchronize(0)
            result['cuda_wall_seconds'] = time.perf_counter()-started
            rows = seq.detach().cpu().tolist()
            if len(rows) != 1: raise ValueError('Expected one physical return')
            evidence = returned_record(input_ids,rows[0],2,cap)
            evidence['decoded_raw_with_special_tokens'] = self.tok.decode(evidence['continuation_ids'],skip_special_tokens=False,clean_up_tokenization_spaces=False)
            evidence['decoded_continuation_untrimmed'] = self.tok.decode(evidence['continuation_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False)
            result.update(completed=True,output=evidence)
            self.costs['generate_completed'] += 1
            self.costs['returned_tokens_through_first_eos'] += evidence['tokens_through_first_eos']
            stats = self.costs['roles'][role]
            stats['completed_calls'] += 1
            stats['output_tokens'] += evidence['tokens_through_first_eos']
            stats['input_tokens'] += len(input_ids)
            stats['cuda_wall_seconds'] += result['cuda_wall_seconds']
            if role != 'feedback': self.costs['generated_slots_actual'] += 1
            return dict(raw_text=evidence['decoded_continuation_untrimmed'],call_index=number,
                        input_tokens=len(input_ids),output_tokens=evidence['tokens_through_first_eos'],
                        cuda_wall_seconds=result['cuda_wall_seconds'],
                        reached_cap_without_eos=evidence['reached_cap_without_eos'],
                        call_started_path=str(start_path),call_result_path=str(self.directory/'calls'/(stem+'_result.json')))
        except BaseException as error:
            result['exception'] = repr(error)
            raise
        finally:
            result.setdefault('cuda_wall_seconds',time.perf_counter()-started)
            self.costs['generate_cuda_wall_seconds'] += result['cuda_wall_seconds']
            result['finished_at_unix'] = time.time()
            write(self.directory/'calls'/(stem+'_result.json'),result,compact=True)

    def memory(self):
        import torch
        self.costs['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(0)
        self.costs['peak_reserved_bytes'] = torch.cuda.max_memory_reserved(0)
