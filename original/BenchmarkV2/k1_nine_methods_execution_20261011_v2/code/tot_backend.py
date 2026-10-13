"""Two-model BF16/unmerged backend, with immutable per-input physical batches."""
import json
import os
import time
from collections import Counter
from tot_protocol import ROOT, MODEL_IDS, read, save, digest


def seed_all(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def trim_tokens(tokens, eos):
    assert tokens
    count = tokens.index(eos)+1 if eos in tokens else len(tokens)
    assert all(t == eos for t in tokens[count:])
    generated = tokens[:count]
    ended = generated[-1] == eos
    return generated, generated[:-1] if ended else generated, ended


class Backend:
    def __init__(self, model, contract, contract_sha, directory):
        self.model_name, self.contract, self.contract_sha = model, contract, contract_sha
        self.directory = directory
        self.model = None

    def start(self):
        if not os.environ.get('SLURM_JOB_ID'):
            raise RuntimeError('GPU generation requires a Slurm allocation')
        import torch, transformers, peft
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        from dataclasses import asdict
        assert (torch.__version__, transformers.__version__, peft.__version__) == ('2.7.0+cu128', '4.46.3', '0.14.0')
        assert torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported()
        torch.cuda.set_device(0)
        torch.set_num_threads(8)
        gpu = torch.cuda.get_device_properties(0)
        assert gpu.total_memory >= 40*1024**3
        spec = self.contract['models'][self.model_name]
        tokenspec = self.contract['tokenizer'][self.model_name]
        self.tok = AutoTokenizer.from_pretrained(spec['base_path'], use_fast=tokenspec['use_fast'],
                                                local_files_only=True, trust_remote_code=False)
        self.tok.pad_token_id = self.tok.eos_token_id
        self.tok.padding_side = 'left'
        self.tok.clean_up_tokenization_spaces = False
        assert self.tok.is_fast == tokenspec['use_fast']
        assert (self.tok.bos_token_id, self.tok.eos_token_id, self.tok.pad_token_id) == (
            tokenspec['bos'], tokenspec['eos'], tokenspec['pad'])
        self.defaults = read(self.contract['generation_defaults_path'])
        seed_all(601270000)
        started = time.perf_counter()
        base = AutoModelForCausalLM.from_pretrained(spec['base_path'], torch_dtype=torch.bfloat16,
                device_map={'': 'cuda:0'}, local_files_only=True, trust_remote_code=False, use_safetensors=True)
        self.model = PeftModel.from_pretrained(base, spec['adapter_path'], adapter_name='default',
                is_trainable=False, autocast_adapter_dtype=True, low_cpu_mem_usage=False,
                ephemeral_gpu_offload=False, local_files_only=True)
        self.model.eval()
        torch.cuda.synchronize()
        self.load_seconds = time.perf_counter()-started
        status = asdict(self.model.get_model_status())
        assert status['enabled'] is True and status['active_adapters'] == ['default'] and status['merged_adapters'] == []
        placements = Counter((str(p.device), str(p.dtype), 'lora' if 'lora_' in n else 'base')
                             for n, p in self.model.named_parameters())
        assert all(dev == 'cuda:0' and dt == ('torch.float32' if kind == 'lora' else 'torch.bfloat16')
                   for dev, dt, kind in placements)
        identity = {'model_id': MODEL_IDS[self.model_name], 'contract_sha256': self.contract_sha,
            'load_spec': spec, 'adapter_status': status, 'training': self.model.training,
            'gpu': {'name': gpu.name, 'memory_bytes': gpu.total_memory},
            'versions': {'torch': torch.__version__, 'transformers': transformers.__version__, 'peft': peft.__version__},
            'tokenizer': dict(tokenspec, padding_side='left', chat_template_applied=False, add_special_tokens=True),
            'parameter_placements': [{'device': k[0], 'dtype': k[1], 'kind': k[2], 'tensors': v} for k, v in placements.items()],
            'attention_implementation': self.model.config._attn_implementation, 'load_seconds': self.load_seconds,
            'slurm_job_id': os.environ['SLURM_JOB_ID']}
        save(self.directory/'model_identity.json', identity)

    def verify_prefix_tokens(self, prefix):
        for call in prefix['initial_provenance']['calls']:
            assert self.tok.encode(call['prompt'], add_special_tokens=True, truncation=False) == call['input_ids']

    def generate(self, prompts, phase, seed, path):
        import torch
        from transformers import GenerationConfig
        expected = 15 if phase == 'children' else 5
        assert len(prompts) == expected
        cap = self.contract['child_max_new_tokens'] if phase == 'children' else self.contract['judge_max_new_tokens']
        limit = self.contract['child_prompt_limit'] if phase == 'children' else self.contract['judge_prompt_limit']
        inputs = [self.tok.encode(p, add_special_tokens=True, truncation=False) for p in prompts]
        assert all(ids and ids[0] == self.tok.bos_token_id and len(ids) <= limit and
                   len(ids)+cap <= self.contract['context_limit'] for ids in inputs), 'Context overflow: no truncation'
        request = {'model': self.model_name, 'contract_sha256': self.contract_sha, 'phase': phase,
                   'seed': seed, 'prompts': prompts, 'input_ids': inputs, 'max_new_tokens': cap}
        if path.exists():
            cached = read(path)
            assert cached['request'] == request and cached['status'] == 'complete'
            return cached
        width = max(map(len, inputs))
        pad = self.tok.pad_token_id
        ids = torch.tensor([[pad]*(width-len(x))+x for x in inputs], dtype=torch.long, device='cuda:0')
        mask = torch.tensor([[0]*(width-len(x))+[1]*len(x) for x in inputs], dtype=torch.long, device='cuda:0')
        config = GenerationConfig.from_dict(self.defaults)
        updates = dict(self.contract['sampling'], max_new_tokens=cap, bos_token_id=self.tok.bos_token_id,
                       eos_token_id=self.tok.eos_token_id, pad_token_id=pad)
        assert not config.update(**updates)
        config.validate()
        started_path = path.with_name(path.stem+'.started.json')
        if started_path.exists():
            assert read(started_path)['request'] == request
        else:
            save(started_path, {'request': request, 'effective_decoding': config.to_dict(),
                 'started_unix': time.time(), 'job': os.environ['SLURM_JOB_ID']})
        seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            sequences = self.model.generate(input_ids=ids, attention_mask=mask,
                generation_config=config, synced_gpus=False).detach().cpu().tolist()
        torch.cuda.synchronize()
        seconds = time.perf_counter()-started
        assert len(sequences) == expected
        padded = ids.cpu().tolist()
        outputs = []
        for row, sequence in enumerate(sequences):
            assert sequence[:width] == padded[row]
            continuation = sequence[width:]
            assert 1 <= len(continuation) <= cap
            generated, content, ended = trim_tokens(continuation, self.tok.eos_token_id)
            outputs.append({'raw_text': self.tok.decode(content, skip_special_tokens=True, clean_up_tokenization_spaces=False),
                'generated_ids': generated, 'content_ids': content, 'input_tokens': len(inputs[row]),
                'output_tokens': len(generated), 'ended_with_eos': ended, 'batch_row': row,
                'post_eos_padding_tokens': len(continuation)-len(generated), 'seed': seed,
                'call_path': str(path), 'cap_exhausted': len(generated) == cap,
                'reached_cap_without_eos': not ended and len(generated) == cap})
        result = {'status': 'complete', 'request': request, 'effective_decoding': config.to_dict(),
                  'outputs': outputs, 'physical_cuda_wall_seconds': seconds,
                  'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'completed_unix': time.time(), 'slurm_job_id': os.environ['SLURM_JOB_ID']}
        save(path, result)
        print(json.dumps({'event': 'tot_batch_complete', 'phase': phase, 'responses': expected,
                          'seconds': round(seconds, 3), 'path': str(path)}), flush=True)
        return result
