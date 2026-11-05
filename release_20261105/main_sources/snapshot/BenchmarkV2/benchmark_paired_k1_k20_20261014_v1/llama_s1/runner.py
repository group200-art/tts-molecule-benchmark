"""Llama s1-style sampled R20; immutable native controller, no property feedback."""
import argparse
import fcntl
import gzip
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
NATIVE = REPO / 'BenchmarkV2/p6_llama_main_ten_tasks_v1'
EXECUTION = 'GELLMO-LLAMA-S1-B512-SAMPLED-R20-20261014-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    with (gzip.open if path.suffix == '.gz' else open)(str(path), 'rt') as stream:
        return json.load(stream)


def save(path, value):
    path = Path(path)
    assert ROOT in path.resolve().parents and path.resolve() == path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.tmp.' + str(os.getpid()))
    payload = (json.dumps(value, ensure_ascii=False, allow_nan=False,
                         separators=(',', ':')) + '\n').encode()
    with tmp.open('xb') as stream:
        if path.suffix == '.gz':
            with gzip.GzipFile(fileobj=stream, mode='wb', mtime=0) as output:
                output.write(payload)
        else:
            stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(str(tmp), str(path))
    tmp.unlink()


def native():
    sys.path.insert(0, str(NATIVE / 'src'))
    c = importlib.import_module('contracts')
    s = importlib.import_module('s1bf')
    p = importlib.import_module('p9_adapters')
    assert c.ROOT == NATIVE and Path(s.__file__).resolve() == NATIVE / 'src/s1bf.py'
    return c, s, p


class NeedCall(Exception):
    def __init__(self, request):
        self.request = request


def trim_generated(tokens, cap, eos=128009):
    valid = tokens[:cap]
    assert valid and len(tokens) >= len(valid)
    count = valid.index(eos) + 1 if eos in valid else len(valid)
    assert all(token == eos for token in tokens[count:])
    generated = valid[:count]
    ended = generated[-1] == eos
    return generated, generated[:-1] if ended else generated, ended


def next_group(requests):
    priorities = {'initial': 0, 'continuation': 1, 'final': 2}
    first = min(requests, key=lambda r: (priorities[r['role']], r['minimum']))
    return [r for r in requests if (priorities[r['role']], r['minimum']) ==
            (priorities[first['role']], first['minimum'])]


def group_seed(batch, requests):
    return int(digest([EXECUTION, batch['batch_id'], [r['key'] for r in requests]])[:15], 16) % 2147483647


class Replay:
    def __init__(self, batch, tokenizer):
        self.batch, self.tok, self.calls = batch, tokenizer, {}
        _, self.s1, self.prompts = native()

    def advance(self, lane):
        ordinal = [0]

        def call(ids, cap, minimum, context):
            request = dict(lane_id=lane['lane_id'], ordinal=ordinal[0], role=context['stage'],
                prompt=None, input_ids=ids, cap=cap, minimum=minimum, context=context)
            ordinal[0] += 1
            request['key'] = digest(request)
            if request['key'] not in self.calls:
                raise NeedCall(request)
            value = self.calls[request['key']]
            assert value['request'] == request
            return value['output']

        class Adapter:
            wait_ids = self.tok.encode('Wait', add_special_tokens=False)
            final_cue_ids = self.tok.encode('\n%%% Response:', add_special_tokens=False)

            def generate_ids(obj, ids, cap, min_new_tokens, context):
                return call(ids, cap, min_new_tokens, context)

        record = lane['record']
        prompt = self.prompts.build_prompt(record)
        ids = self.tok.encode(prompt, add_special_tokens=True, truncation=False)
        trace = self.s1.run_one_budget(Adapter(), ids, 512,
            dict(input_uid=record['input_uid'], repeat=lane['repeat']))
        parsed = self.prompts.parse_final(trace['final_text'])
        ident = dict(method_id='s1-B512-sampled-R20', model_id='gellmo_p6_llama',
            task_id=record['task_id'], input_uid=record['input_uid'],
            original_slot_id=lane['repeat'], execution=EXECUTION)
        terminal = dict(ident, candidate_id='llama-s1-terminal-' + digest(ident),
            input_idx=record['input_idx'], source_smiles=record['source_smiles'],
            raw_smiles=parsed['raw_smiles'], raw_text=trace['final_text'],
            trace=trace, parse=parsed, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
        return dict(lane_id=lane['lane_id'], repeat=lane['repeat'], record=record,
                    terminal=terminal, trajectory=trace)

    def pending(self):
        done, requests = [], []
        for lane in self.batch['lanes']:
            try:
                done.append(self.advance(lane))
            except NeedCall as exc:
                requests.append(exc.request)
        return done, requests


class Model:
    def __init__(self, contract, attempt):
        self.contract, self.attempt = contract, attempt
        self.model = None

    def start(self):
        code = REPO / 'BenchmarkV2/k1_nine_methods_execution_20261011_v2/code'
        sys.path.insert(0, str(code))
        protocol = importlib.import_module('tot_protocol')
        protocol.ROOT = ROOT
        self.tb = importlib.import_module('tot_backend')
        self.backend = self.tb.Backend('llama', self.contract['model_assembly'],
            sha(ROOT / 'contract.json'), self.attempt)
        self.backend.start()
        self.model, self.tok = self.backend.model, self.backend.tok

    def run(self, batch, requests, index, path):
        import torch
        from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList
        assert requests and len(requests) <= 80
        caps = [r['cap'] for r in requests]
        assert len({r['minimum'] for r in requests}) == 1
        assert all(0 < r['cap'] <= 512 and r['input_ids'][0] == 128000 for r in requests)
        width = max(len(r['input_ids']) for r in requests)
        assert width + max(caps) <= self.contract['context_limit']
        inputs = [[128009] * (width - len(r['input_ids'])) + r['input_ids'] for r in requests]
        masks = [[0] * (width - len(r['input_ids'])) + [1] * len(r['input_ids']) for r in requests]
        ids = torch.tensor(inputs, dtype=torch.long, device='cuda:0')
        mask = torch.tensor(masks, dtype=torch.long, device='cuda:0')
        config = GenerationConfig.from_dict(self.backend.defaults)
        update = dict(self.contract['decoding'], max_new_tokens=max(caps),
            min_new_tokens=requests[0]['minimum'], bos_token_id=128000,
            eos_token_id=128009, pad_token_id=128009, return_dict_in_generate=False)
        assert not config.update(**update)
        config.validate()

        class RowCaps(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return input_ids.shape[-1] - width >= torch.tensor(caps, device=input_ids.device)

        seed = group_seed(batch, requests)
        started_path = path.with_suffix('.started.json')
        if not started_path.exists():
            save(started_path, dict(requests=requests, seed=seed,
                effective_decoding=config.to_dict(), row_caps=caps, job=os.environ['SLURM_JOB_ID']))
        else:
            previous = read(started_path)
            assert previous['requests'] == requests and previous['seed'] == seed
        self.tb.seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            sequences = self.model.generate(input_ids=ids, attention_mask=mask,
                generation_config=config, stopping_criteria=StoppingCriteriaList([RowCaps()]),
                synced_gpus=False).detach().cpu().tolist()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        assert len(sequences) == len(requests)
        outputs = []
        for row, (request, sequence, padded) in enumerate(zip(requests, sequences, inputs)):
            assert sequence[:width] == padded
            generated, content, ended = trim_generated(sequence[width:], request['cap'])
            text = self.tok.decode(content, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            output = dict(raw_text=text, text=text, generated_ids=generated, content_ids=content,
                ended_with_eos=ended, output_tokens=len(generated), input_tokens=len(request['input_ids']),
                post_eos_or_cap_padding_tokens=len(sequence)-width-len(generated),
                call_index=index, logical_call_index=request['ordinal'], batch_row=row, seed=seed,
                cuda_wall_seconds=seconds/len(requests), timing_basis='physical batch time / active rows',
                max_new_tokens=request['cap'], min_new_tokens=request['minimum'], length_stop=not ended,
                cap_exhausted=len(generated)==request['cap'],
                reached_cap_without_eos=not ended and len(generated)==request['cap'], call_path=str(path))
            outputs.append(dict(request=request, output=output))
        result = dict(status='complete', batch_id=batch['batch_id'], index=index, seed=seed,
            physical_cuda_wall_seconds=seconds, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), row_caps=caps,
            effective_decoding=config.to_dict(), outputs=outputs, contract_sha256=sha(ROOT/'contract.json'),
            runner_sha256=sha(__file__), job=os.environ['SLURM_JOB_ID'])
        save(path, result)
        print(json.dumps(dict(event='generation', batch=batch['batch_id'], call=index,
            role=requests[0]['role'], lanes=len(outputs), seconds=seconds)), flush=True)
        return result


def run_batch(batch, model, directory):
    path = directory / 'generation.json.gz'
    if path.exists():
        value = read(path)
        assert value['batch'] == batch and value['contract_sha256'] == sha(ROOT/'contract.json')
        assert value['runner_sha256'] == sha(__file__)
        return value
    replay = Replay(batch, model.tok)
    physical = []
    for call_path in sorted((directory/'calls').glob('*.json.gz')):
        value = read(call_path)
        assert value['contract_sha256'] == sha(ROOT/'contract.json')
        assert value['runner_sha256'] == sha(__file__)
        assert value['batch_id'] == batch['batch_id'] and value['index'] == len(physical)
        requests = [entry['request'] for entry in value['outputs']]
        assert value['seed'] == group_seed(batch, requests)
        for entry in value['outputs']:
            key = entry['request']['key']
            assert key not in replay.calls
            replay.calls[key] = entry
        physical.append(value)
    started = time.perf_counter()
    while True:
        done, pending = replay.pending()
        if len(done) == len(batch['lanes']):
            break
        requests = next_group(pending)
        index = len(physical)
        value = model.run(batch, requests, index, directory/'calls'/('{:03d}.json.gz'.format(index)))
        for entry in value['outputs']:
            key = entry['request']['key']
            assert key not in replay.calls
            replay.calls[key] = entry
        physical.append(value)
    assert [v['lane_id'] for v in done] == [v['lane_id'] for v in batch['lanes']]
    value = dict(status='complete', batch=batch, lanes=done, contract_sha256=sha(ROOT/'contract.json'),
        runner_sha256=sha(__file__), physical_calls=len(physical), logical_calls=len(replay.calls),
        physical_cuda_seconds=sum(c['physical_cuda_wall_seconds'] for c in physical),
        generation_wall_seconds_this_attempt=time.perf_counter()-started,
        feedback_batches=0, feedback_candidates=0, properties_evaluated=False)
    save(path, value)
    return value


def verify():
    contract = read(ROOT/'contract.json')
    assert contract['execution_id'] == EXECUTION
    for path, expected in contract['immutable_sources'].items():
        assert sha(path) == expected, path
    for row in contract['model_files_metadata']:
        info = Path(row['path']).stat()
        assert info.st_size == row['bytes'] and info.st_mtime_ns == row['mtime_ns'], row['path']
    assert sha(ROOT/'units.json') == contract['units_sha256']
    return contract, read(ROOT/'units.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--unit-index', type=int)
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    contract, units = verify()
    c, _, _ = native()
    assert sum(u['n_inputs'] for u in units) == 3905
    for unit in units:
        assert digest(c.manifest(unit['task_id'])[unit['start']:unit['stop']]) == unit['records_sha256']
    if args.audit_only:
        print(json.dumps(dict(status='PASS_CPU_CONTRACT_AND_MANIFEST', inputs=3905,
            terminal_slots=78100, units=len(units))))
        return
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('PAIRED_K20_EXECUTION') == EXECUTION
    assert os.environ.get('SLURM_RESTART_COUNT', '0') == '0'
    if args.probe:
        selected = [units[i] for i in contract['probe_unit_indices']]
    else:
        gate = read(ROOT/'engineering_gate.json')
        assert gate['status']=='PASS' and gate['contract_sha256']==sha(ROOT/'contract.json')
        assert gate['runner_sha256']==sha(__file__)
        assert args.unit_index is not None and 0 <= args.unit_index < len(units)
        selected = [units[args.unit_index]]
    attempt = ROOT/'attempts'/(os.environ['SLURM_JOB_ID']+'_'+str(os.getpid()))
    attempt.mkdir(parents=True, exist_ok=False)
    model = Model(contract, attempt)
    started = time.perf_counter()
    results = []
    for unit in selected:
        directory = ROOT/'runs'/unit['unit_id']
        directory.mkdir(parents=True, exist_ok=True)
        lock = (directory/'.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = c.manifest(unit['task_id'])[unit['start']:unit['stop']]
        batches = unit['batches'][:1] if args.probe else unit['batches']
        unit_results = []
        for entry in batches:
            part = rows[entry['offset']:entry['offset']+4]
            assert [r['input_uid'] for r in part] == entry['input_uids']
            batch = dict(batch_id=entry['batch_id'], method='s1', task_id=unit['task_id'],
                records=part, lanes=[dict(lane_id=r['input_uid']+'_r{:02d}'.format(k), record=r, repeat=k)
                for r in part for k in range(20)])
            folder = directory/entry['batch_id']
            if model.model is None and not (folder/'generation.json.gz').exists():
                model.start()
            value = run_batch(batch, model, folder)
            receipt = dict(path=str(folder/'generation.json.gz'), sha256=sha(folder/'generation.json.gz'),
                terminal_slots=len(value['lanes']), generation_seconds=value['physical_cuda_seconds'])
            unit_results.append(receipt)
            results.append(receipt)
        if not args.probe:
            receipt = dict(status='GENERATION_COMPLETE_PROPERTIES_PENDING', unit=unit,
                terminal_slots=sum(r['terminal_slots'] for r in unit_results), paths=unit_results,
                contract_sha256=sha(ROOT/'contract.json'), runner_sha256=sha(__file__))
            assert receipt['terminal_slots']==unit['n_inputs']*20
            if (directory/'complete.json').exists():
                assert read(directory/'complete.json')==receipt
            else:
                save(directory/'complete.json',receipt)
        lock.close()
    save(attempt/'complete.json',dict(status='SELECTED_GENERATION_COMPLETE_PROPERTIES_PENDING',
        probe=args.probe, paths=results, seconds=time.perf_counter()-started,
        contract_sha256=sha(ROOT/'contract.json'), runner_sha256=sha(__file__)))


if __name__ == '__main__':
    main()
