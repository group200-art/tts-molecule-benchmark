"""Mechanism and stopping regressions, not quality-based pilot selection."""
import copy
from common import *
from replay import Replay, next_group, group_seed
from s1bf import run_one_budget
from p9_adapters import build_prompt
from reflexion import run_input
from batched_generation import trim_generated, row_cap_criterion


class Tokenizer:
    def encode(self, text, add_special_tokens=True, **kwargs):
        return ([1] if add_special_tokens else []) + [10 + ord(x) for x in text]


def fake_output(req):
    if req['role'] == 'initial':
        tokens = [11] * 512 if int(req['lane_id'][-2:]) % 3 == 0 else [11, 2]
    elif req['role'] == 'continuation':
        tokens = [12, 2]
    else:
        tokens = [13, 2]
    ended = tokens[-1] == 2
    text = '<SMILES> CCO </SMILES>\n<SMILES> CCN </SMILES>'
    return dict(raw_text=text, text=text, generated_ids=tokens, content_ids=tokens[:-1] if ended else tokens,
                ended_with_eos=ended, input_tokens=len(req['input_ids']), output_tokens=len(tokens),
                call_index=req['ordinal'], max_new_tokens=req['cap'], min_new_tokens=req['minimum'],
                length_stop=not ended, cap_exhausted=len(tokens) == req['cap'], cuda_wall_seconds=0.0)


def fake_outcome(candidate):
    # Repeat classes succeed at trials0/1 or exhaust all3; includes separate memories.
    rep = int(candidate['run_id'].rsplit('_r', 1)[1])
    success = (rep % 3 != 2 and candidate['trial_index'] == rep % 3)
    return dict(feedback=dict(success=success, reward=int(success),
                             observation='Answer is CORRECT' if success else 'Answer is INCORRECT'))


def controller_checks():
    checks = []
    tok = Tokenizer()
    for method in METHODS:
        unit = next(u for u in units() if u['method'] == method)
        batch = next(batches(unit))
        replay = Replay(batch, tok)
        for _ in range(20):
            done, calls, evaluations = replay.pending()
            if len(done) == 80:
                break
            if evaluations:
                for e in evaluations:
                    c = e['candidate']
                    replay.evaluations[c['candidate_id']] = dict(candidate=c, outcome=fake_outcome(c))
            else:
                group = next_group(calls)
                assert group_seed(batch, group) == group_seed(copy.deepcopy(batch), copy.deepcopy(group))
                for req in group:
                    replay.calls[req['key']] = dict(request=req, output=fake_output(req))
        assert len(done) == 80 and len({r['terminal']['candidate_id'] for r in done}) == 80
        for lane, result in zip(batch['lanes'], done):
            saved = sorted([v for v in replay.calls.values() if v['request']['lane_id'] == lane['lane_id']],
                           key=lambda v: v['request']['ordinal'])
            cursor = [0]

            def serial(role, prompt=None, ids=None, cap=250, minimum=0, context=None):
                item = saved[cursor[0]]
                cursor[0] += 1
                req = item['request']
                assert req['role'] == role and req['cap'] == cap and req['minimum'] == minimum
                assert req['prompt'] == prompt and req['input_ids'] == (ids if ids is not None else tok.encode(prompt))
                return item['output']

            r = lane['record']
            if method == 's1':
                class Serial:
                    wait_ids = tok.encode('Wait', add_special_tokens=False)
                    final_cue_ids = tok.encode('\n%%% Response:', add_special_tokens=False)

                    def generate_ids(self, ids, cap, minimum, context):
                        return serial(context['stage'], ids=ids, cap=cap, minimum=minimum, context=context)

                expected = run_one_budget(Serial(), tok.encode(build_prompt(r)), 512,
                                          dict(input_uid=r['input_uid'], repeat=lane['repeat']))
                assert result['trajectory'] == expected and expected['pre_tokens'] <= 512
                assert result['terminal']['raw_smiles'] == 'CCN'
            elif method == 'reflexion':
                expected = run_input(r, EXECUTION + '/' + batch['batch_id'] + '/' + lane['lane_id'], replay.templates,
                    lambda role, prompt, context: serial(role, prompt=prompt, context=context), fake_outcome, lambda *a: None)
                assert expected == result['trajectory']
                assert expected['terminal_trial'] == lane['repeat'] % 3
                assert expected['trials'][0]['memory_before'] == []
                assert result['terminal']['raw_smiles'] == 'CCO'
            else:
                assert len(saved) == 1 and result['terminal']['raw_smiles'] == 'CCN'
        checks.append(method + ':80 lanes match original controller/prompts, full20 slots, correct parser')
    return checks


def stop_checks():
    import torch
    from transformers import MistralConfig, MistralForCausalLM, LogitsProcessor, LogitsProcessorList, StoppingCriteriaList
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = MistralForCausalLM(MistralConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                    num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
                    bos_token_id=1, eos_token_id=2, pad_token_id=2)).eval()
    tokens = torch.tensor([[2, 1, 8], [1, 8, 9], [2, 1, 9]])
    mask = torch.tensor([[0, 1, 1], [1, 1, 1], [0, 1, 1]])

    class ForceToken(LogitsProcessor):
        def __init__(self, eos_at=None):
            self.eos_at = eos_at

        def __call__(self, input_ids, scores):
            scores[:] = -float('inf')
            token = 2 if self.eos_at is not None and input_ids.shape[-1] - 3 >= self.eos_at else 7
            scores[:, token] = 0
            return scores

    for eos_at in (None, 1):
        with torch.inference_mode():
            result = model.generate(tokens, attention_mask=mask, do_sample=False, max_new_tokens=5,
                logits_processor=LogitsProcessorList([ForceToken(eos_at)]),
                stopping_criteria=StoppingCriteriaList([row_cap_criterion([1, 3, 5], 3)]))
        for i, cap in enumerate([1, 3, 5]):
            generated, content, ended = trim_generated(result[i, 3:].tolist(), cap)
            assert result[i, :3].tolist() == tokens[i].tolist()
            assert len(generated) == (cap if eos_at is None else min(2, cap))
            assert ended == (eos_at is not None and cap >= 2)
    assert trim_generated([7, 2, 2], 1) == ([7], [7], False)
    return 'HF4.46.3 tiny Mistral confirms independent row caps, EOS stop, left padding and cap-before-EOS trimming'


def main():
    verify()
    checks = controller_checks()
    checks.append(stop_checks())
    result = dict(status='PASS', checks=checks, plan_sha256=sha(ROOT / 'plan.json'))
    save(ROOT / 'cpu_checks.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
