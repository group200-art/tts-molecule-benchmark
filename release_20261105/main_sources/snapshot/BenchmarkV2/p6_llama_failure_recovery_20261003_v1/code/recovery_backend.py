"""Replay verified GoT call prefixes while restoring both exact RNG streams."""
import hashlib
import pickle
import zlib
from generation import Backend as OriginalBackend, rng_capture, rng_restore
from common import append
from recovery_generation import recovery_unit, verify_source


def load_rng(path, reference):
    # Our own archived RNG data, pinned by SHA256 before the worker runs.
    with path.open('rb') as stream:
        stream.seek(reference['offset'])
        payload = stream.read(reference['compressed_bytes'])
    assert len(payload) == reference['compressed_bytes']
    data = zlib.decompress(payload)
    assert hashlib.sha256(data).hexdigest() == reference['sha256']
    return pickle.loads(data)


class Backend(OriginalBackend):
    def __init__(self, directory, unit):
        super().__init__(directory, unit)
        self.recovery = recovery_unit(directory)
        self.old = verify_source(self.recovery)
        self.saved = []
        import json
        with (self.old/'calls.jsonl').open() as stream:
            for line in stream:
                row = json.loads(line)
                if row['event'] == 'result':
                    assert row['completed'] and row['exception'] is None
                    self.saved.append(row)
        assert len(self.saved) == self.recovery['reusable_calls']
        assert [c['call_index'] for c in self.saved] == list(range(len(self.saved)))
        self.cost.update(reused_calls=0, new_calls=0, reused_generate_seconds=0.0)

    def generate(self, prompt, context, cfg):
        index = self.cost['calls_started']
        if index >= len(self.saved):
            answer = super().generate(prompt, context, cfg)
            self.cost['new_calls'] += 1
            return answer
        call = self.saved[index]
        assert call['prompt'] == prompt, 'Replay prompt differs at call '+str(index)
        assert call['context'] == context, 'Replay context differs at call '+str(index)
        assert call['generation_config_passed'] == cfg.to_dict()
        assert call['base_generation_config'] == self.base_generation.to_dict()
        ids = self.tok(prompt, add_special_tokens=True, truncation=False, return_attention_mask=True)
        assert ids['input_ids'] == call['input_ids'] and ids['attention_mask'] == call['attention_mask']
        before = load_rng(self.old/'rng_states.bin', call['rng_before'])
        assert rng_capture() == before, 'Replay RNG differs at call '+str(index)
        after = load_rng(self.old/'rng_states.bin', call['rng_after'])
        rng_restore(after)
        self.cost['calls_started'] += 1
        self.cost['calls_completed'] += 1
        self.cost['reused_calls'] += 1
        self.cost['reused_generate_seconds'] += call['cuda_wall_seconds']
        g = self.cost['by_G'][str(self.G)]
        g['calls_started'] += 1
        g['calls_completed'] += 1
        g['generate_seconds'] += call['cuda_wall_seconds']
        g['prompt_tokens'] += call['prompt_tokens']
        g['output_tokens_through_eos'] += call['tokens_through_first_eos']
        g['cap_hits'] += int(call['reached_cap_without_eos'])
        append(self.directory/'replayed_calls.jsonl', dict(call_index=index, context=context,
               source=str(self.old/'calls.jsonl'), rng_before=call['rng_before'],
               rng_after=call['rng_after'], prompt_and_tokens_verified=True,
               rng_verified=True, generated_now=False))
        return call['decoded_continuation_untrimmed']
