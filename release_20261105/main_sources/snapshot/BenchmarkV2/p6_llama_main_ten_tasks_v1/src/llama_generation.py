"""One unmerged author P6-Llama assembly; explicit inherited method decoding."""
import gzip
import json
import time
from collections import Counter
from contracts import ROOT,read,write,append,digest
from runtime_common import require_allocation


def set_seeds(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed(seed);torch.cuda.manual_seed_all(seed)


class Generation:
    def __init__(self,method,cfg,costs,directory,maximum_calls):
        self.method,self.cfg,self.costs,self.d,self.maximum_calls=method,cfg,costs,directory,maximum_calls
        self.model=None

    def start(self):
        require_allocation()
        import torch,transformers,peft
        from transformers import AutoTokenizer,AutoModelForCausalLM
        from peft import PeftModel
        from dataclasses import asdict
        assert self.model is None
        assert (torch.__version__,transformers.__version__,peft.__version__)==('2.7.0+cu128','4.46.3','0.14.0')
        assert torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported()
        torch.cuda.set_device(0);torch.set_num_threads(8)
        gpu=torch.cuda.get_device_properties(0)
        assert gpu.name=='NVIDIA RTX PRO 6000 Blackwell Server Edition' and gpu.total_memory>80*1024**3
        set_seeds(self.cfg['seed'])
        spec=read(ROOT/'configs/runtime.json')
        self.tok=AutoTokenizer.from_pretrained(spec['base_path'],use_fast=True,local_files_only=True,trust_remote_code=False)
        self.tok.padding_side=self.cfg['padding_side'];self.tok.pad_token_id=self.tok.eos_token_id
        self.tok.clean_up_tokenization_spaces=False
        assert self.tok.is_fast and (self.tok.bos_token_id,self.tok.eos_token_id,self.tok.pad_token_id)==(128000,128009,128009)
        assert self.tok.encode('molecule',add_special_tokens=True)==[128000]+self.tok.encode('molecule',add_special_tokens=False)
        self.defaults=read(ROOT/'configs/mistral_generation_defaults.json')
        self.wait_ids=self.tok.encode('Wait',add_special_tokens=False)
        self.final_cue_ids=self.tok.encode('\n%%% Response:',add_special_tokens=False)
        identity=dict(model_id='gellmo_p6_llama',load_spec=spec,gpu=dict(name=gpu.name,total_memory_bytes=gpu.total_memory,capability=[gpu.major,gpu.minor]),
            versions=dict(torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__),
            tokenizer=dict(class_name=type(self.tok).__name__,is_fast=True,padding_side=self.tok.padding_side,bos=128000,eos=128009,pad=128009,size=len(self.tok),
                chat_template_applied=False,add_special_tokens=True,clean_up_tokenization_spaces=False),
            startup_seed=self.cfg['seed'],method_main=self.cfg['main_id'],
            default_basis='Original Mistral GenerationConfig defaults, with native Llama BOS/EOS/pad and explicit existing method overrides; no Llama sampling-default leakage')
        write(self.d/'initialization.json',identity)
        self.costs['load_started']=1;write(self.d/'load_started.json',dict(unix=time.time()))
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter()
        base=AutoModelForCausalLM.from_pretrained(spec['base_path'],torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},local_files_only=True,trust_remote_code=False,use_safetensors=True)
        self.model=PeftModel.from_pretrained(base,spec['adapter_path'],adapter_name='default',is_trainable=False,autocast_adapter_dtype=True,
                                            low_cpu_mem_usage=False,ephemeral_gpu_offload=False,local_files_only=True)
        self.model.eval();torch.cuda.synchronize()
        self.costs['load_completed']=1;self.costs['load_cuda_wall_seconds']=time.perf_counter()-t
        status=asdict(self.model.get_model_status())
        assert status['enabled'] is True and status['active_adapters']==['default'] and status['merged_adapters']==[]
        placements=Counter((str(p.device),str(p.dtype),'lora' if 'lora_' in n else 'base') for n,p in self.model.named_parameters())
        assert all(dev=='cuda:0' and dt==('torch.float32' if kind=='lora' else 'torch.bfloat16') for dev,dt,kind in placements)
        identity.update(adapter_status=status,parameter_placements=[dict(device=k[0],dtype=k[1],kind=k[2],tensors=v) for k,v in placements.items()],
            attention_implementation=self.model.config._attn_implementation,training=self.model.training)
        write(self.d/'model_identity.json',identity)

    def generate(self,prompt,context,seed=None,decoding=None,input_ids=None):
        import torch
        from transformers import GenerationConfig
        assert self.costs['generate_started']<self.maximum_calls
        if seed is not None:
            assert self.method in ('p6_tot20','self_refine','ev','fewshot')
            set_seeds(seed)
        spec=dict(self.cfg['decoding'] if decoding is None else decoding)
        cap=spec['max_new_tokens']
        config=GenerationConfig.from_dict(self.defaults)
        spec.update(bos_token_id=128000,eos_token_id=128009,pad_token_id=128009)
        spec.setdefault('return_dict_in_generate',False)
        assert not config.update(**spec);config.validate()
        if input_ids is None:input_ids=self.tok.encode(prompt,add_special_tokens=True,truncation=False)
        else:assert self.method=='s1' and prompt is None
        assert input_ids and input_ids[0]==128000 and len(input_ids)+cap<=self.cfg['context_limit']
        assert len(input_ids)<=self.cfg['prompt_limit']
        callnum=self.costs['generate_started']
        call=dict(call_index=callnum,context=context,seed=seed,prompt=prompt,input_ids=input_ids,
            input_tokens=len(input_ids),prompt_digest=digest(prompt) if prompt is not None else None,
            effective_decoding=config.to_dict(),started=True,completed=False,exception=None)
        path=self.d/'calls'/('{:05d}.json.gz'.format(callnum));path.parent.mkdir(parents=True,exist_ok=True)
        append(self.d/'call_events.jsonl',dict(event='started',call_index=callnum,context=context,seed=seed,path=str(path),unix=time.time()))
        self.costs['generate_started']+=1
        tokens=torch.tensor([input_ids],dtype=torch.long,device='cuda:0')
        torch.cuda.synchronize();started=time.perf_counter()
        try:
            with torch.inference_mode():
                result=self.model.generate(input_ids=tokens,attention_mask=torch.ones_like(tokens),generation_config=config,synced_gpus=False)
            torch.cuda.synchronize();elapsed=time.perf_counter()-started
            sequences=(result.sequences if hasattr(result,'sequences') else result).detach().cpu().tolist()
            call['continuation_ids']=[s[len(input_ids):] for s in sequences]
            assert len(sequences)==config.num_return_sequences and all(s[:len(input_ids)]==input_ids for s in sequences)
            outputs=[]
            for seq in sequences:
                cont=seq[len(input_ids):];assert 1<=len(cont)<=cap
                count=cont.index(128009)+1 if 128009 in cont else len(cont)
                assert all(x==128009 for x in cont[count:])
                generated=cont[:count];ended=generated[-1]==128009
                content=generated[:-1] if ended else generated
                outputs.append(dict(raw_text=self.tok.decode(cont,skip_special_tokens=True,clean_up_tokenization_spaces=False),
                    generated_ids=generated,content_ids=content,ended_with_eos=ended,output_tokens=count,input_tokens=len(input_ids),
                    post_eos_padding_tokens=len(cont)-count,call_index=callnum,cuda_wall_seconds=elapsed,
                    max_new_tokens=cap,min_new_tokens=config.min_new_tokens or 0,length_stop=not ended,
                    cap_exhausted=len(generated)==cap,reached_cap_without_eos=len(generated)==cap and not ended,
                    call_path=str(path)))
            if hasattr(result,'sequences_scores') and result.sequences_scores is not None:
                call['sequences_scores']=result.sequences_scores.detach().cpu().tolist()
            if hasattr(result,'beam_indices') and result.beam_indices is not None:
                call['beam_indices']=result.beam_indices.detach().cpu().tolist()
            call.update(completed=True,cuda_wall_seconds=elapsed,outputs=outputs)
            self.costs['generate_completed']+=1
            self.costs['physical_return_sequences']+=len(outputs)
            self.costs['prompt_tokens_unexpanded']+=len(input_ids)
            self.costs['returned_tokens_through_first_eos']+=sum(o['output_tokens'] for o in outputs)
            self.costs['generate_cuda_wall_seconds']+=elapsed
            return outputs
        except BaseException as err:
            call.update(exception=repr(err),cuda_wall_seconds=time.perf_counter()-started)
            raise
        finally:
            with path.open('xb') as f:
                with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(json.dumps(call,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode())
            append(self.d/'call_events.jsonl',dict(event='result',call_index=callnum,completed=call['completed'],exception=call['exception'],unix=time.time(),path=str(path)))

    def generate_ids(self,ids,cap,min_new_tokens,context):
        assert self.method=='s1'
        spec=dict(self.cfg['decoding'],max_new_tokens=cap,min_new_tokens=min_new_tokens)
        out=self.generate(None,context,decoding=spec,input_ids=ids)[0]
        out['text']=self.tok.decode(out['content_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False)
        return out
