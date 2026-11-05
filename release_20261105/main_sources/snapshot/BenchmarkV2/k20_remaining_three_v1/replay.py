"""Suspend/replay the unchanged serial s1 and Reflexion controllers for batching."""
from common import *
from s1bf import run_one_budget
from p9_adapters import build_prompt, parse_final
from pplus_parser import parse_smiles
from reflexion import run_input


class NeedCall(Exception):
    def __init__(self, request):
        self.request = request


class NeedEvaluation(Exception):
    def __init__(self, candidate):
        self.candidate = candidate


class Replay:
    def __init__(self, batch, tokenizer):
        self.batch = batch
        self.tok = tokenizer
        self.calls = {}
        self.evaluations = {}
        self.templates = read(LEGACY / 'configs/reflexion_prompts.json')
        self.fewshot = None
        if batch['method'] == 'fewshot':
            data = read(LEGACY / 'configs' / ('pplus_prompts_' + batch['task_id'] + '.json'))
            self.fewshot = {p['record']['input_uid']: p for p in data}

    def advance(self, lane):
        method = self.batch['method']
        counter = [0]

        def call(role, prompt=None, ids=None, cap=250, minimum=0, context=None):
            ordinal = counter[0]
            counter[0] += 1
            if ids is None:
                ids = self.tok.encode(prompt, add_special_tokens=True, truncation=False)
            req = dict(lane_id=lane['lane_id'], ordinal=ordinal, role=role, prompt=prompt,
                       input_ids=ids, cap=cap, minimum=minimum, context=context or {})
            key = digest(req)
            req['key'] = key
            if key not in self.calls:
                raise NeedCall(req)
            value = self.calls[key]
            assert value['request'] == req
            return value['output']

        r = lane['record']
        if method == 's1':
            class Adapter:
                wait_ids = self.tok.encode('Wait', add_special_tokens=False)
                final_cue_ids = self.tok.encode('\n%%% Response:', add_special_tokens=False)

                def generate_ids(obj, ids, cap, min_new_tokens, context):
                    return call(context['stage'], ids=ids, cap=cap, minimum=min_new_tokens, context=context)

            ids = self.tok.encode(build_prompt(r), add_special_tokens=True, truncation=False)
            trace = run_one_budget(Adapter(), ids, 512, dict(input_uid=r['input_uid'], repeat=lane['repeat']))
            parsed = parse_final(trace['final_text'])
            c = terminal(method, lane, parsed['raw_smiles'], trace['final_text'], trace)
        elif method == 'fewshot':
            prompt_plan = self.fewshot[r['input_uid']]
            assert prompt_plan['record'] == r and prompt_plan['shots'] == 4
            assert hashlib.sha256(prompt_plan['prompt'].encode()).hexdigest() == prompt_plan['prompt_sha256']
            out = call('answer', prompt=prompt_plan['prompt'], cap=100)
            parsed = parse_smiles(out['raw_text'])
            c = terminal(method, lane, parsed[0], out['raw_text'],
                         dict(answer=out, parse=parsed, prompt_sha256=prompt_plan['prompt_sha256'],
                              demo_uids=prompt_plan['demo_uids']))
            trace = c['trace']
        else:
            assert method == 'reflexion'

            def model_call(role, prompt, context):
                return call(role, prompt=prompt, context=context)

            def evaluate(c):
                key = c['candidate_id']
                if key not in self.evaluations:
                    raise NeedEvaluation(c)
                value = self.evaluations[key]
                assert value['candidate'] == c
                return value['outcome']

            trace = run_input(r, EXECUTION + '/' + self.batch['batch_id'] + '/' + lane['lane_id'],
                              self.templates, model_call, evaluate, lambda *args: None)
            old = trace['native_candidate']
            c = terminal(method, lane, old['raw_smiles'], old['raw_text'],
                         dict(terminal_trial=trace['terminal_trial'], stop_reason=trace['stop_reason'],
                              native_candidate_id=old['candidate_id']))
        return dict(lane_id=lane['lane_id'], repeat=lane['repeat'], record=r, terminal=c, trajectory=trace)

    def pending(self):
        done, calls, evaluations = [], [], []
        for lane in self.batch['lanes']:
            try:
                done.append(self.advance(lane))
            except NeedCall as exc:
                calls.append(exc.request)
            except NeedEvaluation as exc:
                evaluations.append(dict(record=lane['record'], candidate=exc.candidate))
        return done, calls, evaluations


def next_group(calls):
    # Merge all final s1 lanes only after finishing eligible Wait continuations.
    priorities = {'initial': 0, 'continuation': 1, 'final': 2,
                  'answer': 0, 'thought': 0, 'action': 1, 'reflection': 2}
    key = lambda r: (r['context'].get('trial_index', 0), priorities[r['role']], r['minimum'])
    first = min(calls, key=key)
    return [r for r in calls if key(r) == key(first)]


def group_seed(batch, requests):
    return int(digest([EXECUTION, batch['batch_id'], [r['key'] for r in requests]])[:15], 16) % 2147483647
