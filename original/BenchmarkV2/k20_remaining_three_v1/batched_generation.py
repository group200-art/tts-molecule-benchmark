"""Exact token-prefix preservation, native sampling, per-row continuation caps."""
import time
from common import *
from replay import group_seed


def row_cap_criterion(caps, prefix_length):
    import torch
    from transformers import StoppingCriteria

    class RowCaps(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return input_ids.shape[-1] - prefix_length >= torch.tensor(caps, device=input_ids.device)

    return RowCaps()


def trim_generated(tokens, cap):
    # Finished rows receive pad=EOS; exclude that padding BEFORE EOS detection.
    valid = tokens[:cap]
    assert valid
    count = valid.index(2) + 1 if 2 in valid else len(valid)
    assert all(t == 2 for t in tokens[count:])
    generated = valid[:count]
    ended = generated[-1] == 2
    return generated, generated[:-1] if ended else generated, ended


class BatchedGeneration:
    def __init__(self, method, directory):
        self.method = method
        self.directory = directory
        self.model = None

    def start(self):
        from generation import Generation
        import generation
        # Reuse the unchanged weight/tokenizer assembly; redirect only its scoped IO/gate.
        generation.write = save
        generation.require_allocation = require_allocation
        cfg = dict(plan()['extensions'][self.method], padding_side='left')
        costs = dict(load_started=0, load_completed=0)
        loader = Generation(self.method, cfg, costs, self.directory, 1000000)
        loader.start()
        self.model, self.tok, self.defaults = loader.model, loader.tok, loader.defaults
        self.load_seconds = costs['load_cuda_wall_seconds']

    def run(self, batch, requests, index, path):
        import torch
        from transformers import GenerationConfig, StoppingCriteriaList
        from generation import set_seeds
        assert requests and len(requests) <= 80
        caps = [r['cap'] for r in requests]
        assert len({r['minimum'] for r in requests}) == 1
        ids = [r['input_ids'] for r in requests]
        width = max(map(len, ids))
        assert all(x and x[0] == 1 for x in ids)
        assert width + max(caps) <= 32768
        tokens = torch.tensor([[2] * (width - len(x)) + x for x in ids], dtype=torch.long, device='cuda:0')
        mask = torch.tensor([[0] * (width - len(x)) + [1] * len(x) for x in ids], dtype=torch.long, device='cuda:0')
        config = GenerationConfig.from_dict(self.defaults)
        decoding = dict(plan()['extensions'][self.method]['decoding'], max_new_tokens=max(caps),
                        min_new_tokens=requests[0]['minimum'], bos_token_id=1, eos_token_id=2, pad_token_id=2,
                        return_dict_in_generate=False)
        assert not config.update(**decoding)
        config.validate()
        seed = group_seed(batch, requests)
        set_seeds(seed)
        started = time.time()
        save(path.with_suffix('.started.json'), dict(requests=requests, seed=seed, started_unix=started,
             job=os.environ['SLURM_JOB_ID'], effective_decoding=config.to_dict(), row_caps=caps))
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.inference_mode():
            result = self.model.generate(input_ids=tokens, attention_mask=mask, generation_config=config,
                         stopping_criteria=StoppingCriteriaList([row_cap_criterion(caps, width)]), synced_gpus=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t
        sequences = result.detach().cpu().tolist()
        assert len(sequences) == len(requests)
        outputs = []
        for row, (req, sequence, padded) in enumerate(zip(requests, sequences, tokens.cpu().tolist())):
            assert sequence[:width] == padded
            generated, content, ended = trim_generated(sequence[width:], req['cap'])
            text = self.tok.decode(content, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            out = dict(raw_text=text, text=text, generated_ids=generated, content_ids=content, ended_with_eos=ended,
                       output_tokens=len(generated), input_tokens=len(req['input_ids']),
                       post_eos_or_cap_padding_tokens=len(sequence) - width - len(generated),
                       call_index=index, logical_call_index=req['ordinal'], batch_row=row, seed=seed,
                       cuda_wall_seconds=elapsed / len(requests), timing_basis='allocated physical batch time / active rows',
                       max_new_tokens=req['cap'], min_new_tokens=req['minimum'], length_stop=not ended,
                       cap_exhausted=len(generated) == req['cap'], reached_cap_without_eos=not ended and len(generated) == req['cap'],
                       call_path=str(path))
            outputs.append(dict(request=req, output=out))
        result = dict(status='complete', batch_id=batch['batch_id'], index=index, seed=seed,
                      started_unix=started, completed_unix=time.time(), physical_cuda_wall_seconds=elapsed,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(), row_caps=caps,
                      effective_decoding=config.to_dict(), outputs=outputs, plan_sha256=sha(ROOT / 'plan.json'))
        save(path, result)
        print(json.dumps(dict(event='generation', batch=batch['batch_id'], call=index, role=requests[0]['role'],
                              lanes=len(outputs), seconds=round(elapsed, 2)), ensure_ascii=False), flush=True)
        return result
