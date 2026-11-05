"""Exact archived-call replay followed by the original explicit-seed ToT suffix."""
import gzip
import json
from pathlib import Path
from contracts import ROOT, read, write, append, verify
from llama_generation import Generation as OriginalGeneration


def recovery_unit(directory):
    plan = read(ROOT/'configs/recovery.json')
    return next(u for u in plan['units'] if u['unit_id'] == directory.name)


def verify_source(unit):
    for item in unit['source_files']:
        verify(item['path'], item['sha256'])
    old = Path(unit['old_directory'])
    assert read(old/'status.json')['status'] == 'failed'
    return old


class Generation(OriginalGeneration):
    def __init__(self, method, cfg, costs, directory, maximum_calls):
        super().__init__(method, cfg, costs, directory, maximum_calls)
        self.unit = recovery_unit(directory)
        self.old = verify_source(self.unit)
        self.replay_count = self.unit['reusable_calls']
        self.costs.update(reused_generation_calls=0, new_generation_calls=0,
                          reused_generation_cuda_seconds=0.0, new_model_loads=0)

    def start(self):
        # Sampling needs no GPU or model assembly; ToT loads only at the suffix.
        old_cost = read(self.old/'costs.json')
        self.costs.update(load_started=1, load_completed=1,
                          load_cuda_wall_seconds=old_cost['load_cuda_wall_seconds'])
        write(self.d/'generation_recovery.json', dict(source_directory=str(self.old),
              replay_calls=self.replay_count, new_call_limit=self.maximum_calls-self.replay_count,
              sampling_model_load_forbidden=self.method=='sampling40_prefix20'))

    def generate(self, prompt, context, seed=None, decoding=None, input_ids=None):
        index = self.costs['generate_started']
        if index >= self.replay_count:
            assert self.method == 'p6_tot20' and seed is not None
            if self.model is None:
                super().start()
                self.costs['new_model_loads'] += 1
            result = super().generate(prompt, context, seed, decoding, input_ids)
            self.costs['new_generation_calls'] += 1
            return result
        source = self.old/'calls'/('{:05d}.json.gz'.format(index))
        with gzip.open(str(source), 'rt') as stream:
            call = json.load(stream)
        assert call['call_index'] == index and call['completed'] and call['exception'] is None
        assert call['prompt'] == prompt and call['context'] == context and call['seed'] == seed
        assert decoding is None and input_ids is None
        assert all(call['effective_decoding'][k] == v for k, v in self.cfg['decoding'].items())
        assert call['effective_decoding']['eos_token_id'] == 128009
        outputs = call['outputs']
        assert len(outputs) == self.cfg['decoding']['num_return_sequences']
        append(self.d/'replayed_calls.jsonl', dict(call_index=index, source=str(source),
               context=context, seed=seed, outputs=len(outputs), prompt_verified=True,
               decoding_verified=True, generated_now=False))
        self.costs['generate_started'] += 1
        self.costs['generate_completed'] += 1
        self.costs['physical_return_sequences'] += len(outputs)
        self.costs['prompt_tokens_unexpanded'] += call['input_tokens']
        self.costs['returned_tokens_through_first_eos'] += sum(o['output_tokens'] for o in outputs)
        self.costs['generate_cuda_wall_seconds'] += call['cuda_wall_seconds']
        self.costs['reused_generation_cuda_seconds'] += call['cuda_wall_seconds']
        self.costs['reused_generation_calls'] += 1
        return outputs
