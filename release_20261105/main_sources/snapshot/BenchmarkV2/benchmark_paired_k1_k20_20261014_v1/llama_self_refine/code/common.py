"""Independent Llama SR-r1 scope; original task prompts and parser are imported read-only."""
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
LLAMA = REPO / 'BenchmarkV2/p6_llama_main_ten_tasks_v1'
K1 = REPO / 'BenchmarkV2/k1_nine_methods_execution_20261011_v2'
MISTRAL = REPO / 'BenchmarkV2/self_refine_r1_k20_v1'
ICL = REPO / 'BenchmarkV2/p10_n20_r20_two_models_20261013_v1/next_k20/llama_icl_sampled'
sys.path.insert(0, str(LLAMA / 'src'))
from contracts import manifest, task_contract
from self_refine import prompt_for, task_text, parse_answer, call_seed, candidate as legacy_candidate

TASKS = ('BDP', 'BDQ', 'BPQ', 'DPQ', 'BDPQ', 'MPQ', 'BDMQ', 'BHMQ', 'BMPQ', 'HMPQ')
ROLES = [('init', 0), ('feedback', 0), ('refine', 1), ('feedback', 1)]
NEW_SEEDS = list(range(51002, 51021))
EXECUTION = 'GELLMO-LLAMA-SELFREFINE-R1-R20-20261014-v1'


def read(path):
    path = Path(path)
    with (gzip.open if path.suffix == '.gz' else open)(str(path), 'rt', encoding='utf-8') as f:
        return json.load(f)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def save(path, value):
    path = Path(path).absolute()
    assert ROOT in path.parents and path.resolve() == path, str(path)
    if path.exists():
        assert read(path) == value, 'Immutable existing record differs: ' + str(path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.tmp.' + str(os.getpid()))
    payload = (json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n').encode()
    with tmp.open('xb') as f:
        if path.suffix == '.gz':
            with gzip.GzipFile(fileobj=f, mode='wb', mtime=0) as z:
                z.write(payload)
        else:
            f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(str(tmp), str(path))
    finally:
        tmp.unlink()


def verify():
    frozen = read(ROOT / 'reports/freeze.json')
    for path, expected in frozen['sha256'].items():
        assert sha(path) == expected, path
    for item in read(LLAMA / 'evidence/model_stat_inventory.json'):
        st = Path(item['path']).stat()
        assert st.st_size == item['size'] and st.st_mtime_ns == item['mtime_ns'], item['path']
    return read(ROOT / 'contract.json')


def ordered_lanes(batch):
    return [(record, seed) for record in batch['records'] for seed in NEW_SEEDS]


def batch_seed(batch, role, r):
    assert (role, r) in ROLES
    return int(digest([EXECUTION, batch['batch_id'],
                      [(x['task_id'], x['input_uid'], s) for x, s in ordered_lanes(batch)],
                      role, r])[:15], 16) % 2147483647


def call_identity(batch, role, r, prompts):
    return dict(execution_id=EXECUTION, contract_sha256=sha(ROOT / 'contract.json'),
                batch_id=batch['batch_id'], role=role, r=r, seed=batch_seed(batch, role, r),
                lane_keys=[[x['task_id'], x['input_uid'], s] for x, s in ordered_lanes(batch)],
                prompt_digests=[digest(p) for p in prompts])


def audit_batch(result, batch, templates):
    assert result['status'] == 'GENERATION_COMPLETE_PROPERTIES_PENDING'
    assert result['batch_id'] == batch['batch_id']
    assert result['contract_sha256'] == sha(ROOT / 'contract.json')
    assert result['inference_property_calls'] == 0
    assert len(result['lanes']) == len(ordered_lanes(batch))
    for row, (lane, (record, seed)) in enumerate(zip(result['lanes'], ordered_lanes(batch))):
        assert lane['record'] == record and lane['rep_seed'] == seed
        assert lane['repeat'] == seed - 51001
        assert len(lane['answers']) == len(lane['feedbacks']) == 2
        answers, feedbacks = [], []
        for role, r in ROLES:
            call = lane['feedbacks'][r] if role == 'feedback' else lane['answers'][r]
            expected = prompt_for(role, task_text(record), answers, feedbacks, r, templates)
            assert call['prompt_digest'] == digest(expected)
            assert (call['role'], call['r']) == (role, r)
            assert call['batch_row'] == row and call['batch_id'] == batch['batch_id']
            assert call['seed'] == batch_seed(batch, role, r)
            assert call['rep_seed'] == seed
            (feedbacks if role == 'feedback' else answers).append(call)
        assert lane['terminal_parse'] == parse_answer(answers[1]['raw_text'])


def execute_batch(batch, generate, templates):
    lanes = [dict(record=record, rep_seed=seed, repeat=seed-51001, answers=[], feedbacks=[])
             for record, seed in ordered_lanes(batch)]
    measurements = []
    for role, r in ROLES:
        prompts = [prompt_for(role, task_text(x['record']), x['answers'], x['feedbacks'], r, templates)
                   for x in lanes]
        outputs, measurement = generate(role, r, prompts, batch)
        assert len(outputs) == len(lanes)
        for lane, output in zip(lanes, outputs):
            lane['feedbacks' if role == 'feedback' else 'answers'].append(output)
        measurements.append(measurement)
    for lane in lanes:
        lane['terminal_parse'] = parse_answer(lane['answers'][1]['raw_text'])
    result = dict(status='GENERATION_COMPLETE_PROPERTIES_PENDING', batch_id=batch['batch_id'],
                  contract_sha256=sha(ROOT / 'contract.json'), lanes=lanes,
                  measurements=measurements, inference_property_calls=0)
    audit_batch(result, batch, templates)
    return result
