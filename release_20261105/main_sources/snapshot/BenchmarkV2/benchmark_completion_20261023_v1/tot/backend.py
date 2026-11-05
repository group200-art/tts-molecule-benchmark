"""Reuse audited model assembly; explicit child versus judge budgets."""
import importlib
import os
import time
from pathlib import Path
import protocol as p

p.old.ROOT = p.ROOT
native = importlib.import_module('tot_backend')
p.old.ROOT = p.OLD
native.save = p.save


class Backend(native.Backend):
    def generate(self, prompts, phase, seed, path):
        import torch
        from transformers import GenerationConfig
        path = Path(path)
        assert phase in ('missing_children', 'terminal_vote', 'development')
        assert len(prompts) == 5
        is_child = phase == 'missing_children'
        cap = self.contract['child_max_new_tokens'] if is_child else self.contract['judge_max_new_tokens']
        limit = self.contract['child_prompt_limit'] if is_child else self.contract['judge_prompt_limit']
        inputs = [self.tok.encode(text, add_special_tokens=True, truncation=False) for text in prompts]
        assert all(ids and ids[0] == self.tok.bos_token_id and len(ids) <= limit and
                   len(ids) + cap <= self.contract['context_limit'] for ids in inputs), 'Context overflow; no truncation'
        request = dict(model=self.model_name, contract_sha256=self.contract_sha, phase=phase,
                       seed=seed, prompts=prompts, input_ids=inputs, max_new_tokens=cap)
        if path.exists():
            result = p.read(path)
            assert result['status'] == 'complete' and result['request'] == request
            return result
        width, pad = max(map(len, inputs)), self.tok.pad_token_id
        padded = [[pad] * (width - len(x)) + x for x in inputs]
        ids = torch.tensor(padded, dtype=torch.long, device='cuda:0')
        mask = torch.tensor([[0] * (width - len(x)) + [1] * len(x) for x in inputs], dtype=torch.long, device='cuda:0')
        config = GenerationConfig.from_dict(self.defaults)
        assert not config.update(**dict(self.contract['sampling'], max_new_tokens=cap,
            bos_token_id=self.tok.bos_token_id, eos_token_id=self.tok.eos_token_id, pad_token_id=pad))
        config.validate()
        started_path = path.with_suffix('.started.json')
        if started_path.exists():
            assert p.read(started_path)['request'] == request
        else:
            p.save(started_path, dict(request=request, effective_decoding=config.to_dict(),
                                     started_unix=time.time(), job=os.environ['SLURM_JOB_ID']))
        native.seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            sequences = self.model.generate(input_ids=ids, attention_mask=mask,
                generation_config=config, synced_gpus=False).detach().cpu().tolist()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        assert len(sequences) == 5
        outputs = []
        for row, sequence in enumerate(sequences):
            assert sequence[:width] == padded[row]
            continuation = sequence[width:]
            assert 1 <= len(continuation) <= cap
            tokens, content, ended = native.trim_tokens(continuation, self.tok.eos_token_id)
            outputs.append(dict(raw_text=self.tok.decode(content, skip_special_tokens=True,
                clean_up_tokenization_spaces=False), generated_ids=tokens, content_ids=content,
                input_tokens=len(inputs[row]), output_tokens=len(tokens), ended_with_eos=ended,
                batch_row=row, post_eos_padding_tokens=len(continuation) - len(tokens), seed=seed,
                call_path=str(path), cap_exhausted=len(tokens) == cap,
                reached_cap_without_eos=not ended and len(tokens) == cap))
        result = dict(status='complete', request=request, effective_decoding=config.to_dict(),
                      outputs=outputs, physical_cuda_wall_seconds=seconds,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      completed_unix=time.time(), slurm_job_id=os.environ['SLURM_JOB_ID'])
        p.save(path, result)
        return result
