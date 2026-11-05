"""Frozen C20 ToT continuation: parser-only repair; no property access in search."""
import gzip
import hashlib
import importlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BV = ROOT.parents[1]
REPO = BV.parent
OLD = BV / 'k1_nine_methods_execution_20261011_v2'
sys.path.insert(0, str(OLD / 'code'))
old = importlib.import_module('tot_protocol')
TASKS, MODEL_IDS = old.TASKS, old.MODEL_IDS
EXECUTION = 'TOT-B4-C20-PARSER-REPAIR-20261023-v1'
METHOD = 'tot_lm_vote_b4_c20'
SAFE = re.compile(r'\bThe best choice is\s+([0-9]+)(?![\w]|[.]\d|/\d)', re.I)
TAIL = re.compile(r'^\s*(?:,\s*(?:(?:or|and)\s+)?|(?:or|and|and/or)\b\s*|[&/\-–—]\s*)(?:choice\s*)?[0-9]+(?![\w]|[.]\d|/\d)', re.I)


def read(path):
    path = Path(path)
    with (gzip.open if path.suffix == '.gz' else open)(str(path), 'rt') as stream:
        return json.load(stream)


digest, sha = old.digest, old.sha


def save(path, value):
    path = Path(path).resolve()
    assert ROOT in path.parents
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert read(path) == value, ('Immutable result mismatch', str(path))
        return
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'wb') as stream:
            data = (json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n').encode()
            if path.suffix == '.gz':
                with gzip.GzipFile(fileobj=stream, mode='wb', mtime=0) as compressed:
                    compressed.write(data)
            else:
                stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(name, str(path))
    finally:
        os.unlink(name)


def parse_vote(text, n):
    text = str(text)
    matches = list(SAFE.finditer(text))
    numbers = sorted({int(match[1]) for match in matches})
    reason = ('missing' if not numbers else 'ambiguous_number_list' if any(
        TAIL.match(text[match.end():]) for match in matches) else
        'conflicting_ids' if len(numbers) != 1 else
        'out_of_range' if not 1 <= numbers[0] <= n else 'valid')
    return dict(choice_index=numbers[0] - 1 if reason == 'valid' else None,
                reason=reason, explicit_ids=numbers)


def ranking(outputs, n):
    assert n in (5, 20) and len(outputs) == 5
    votes = [parse_vote(out['raw_text'], n) for out in outputs]
    counts = [0] * n
    for vote in votes:
        if vote['choice_index'] is not None:
            counts[vote['choice_index']] += 1
    order = sorted(range(n), key=lambda i: (-counts[i], i))
    return dict(votes=votes, counts=counts, ranking=order, valid_votes=sum(counts),
                all_invalid_fallback=sum(counts) == 0,
                top_tied=sum(count == max(counts) for count in counts) > 1,
                tie_rule='original_frontier_index_ascending')


def judge_prompt(record, objectives, candidates):
    assert len(candidates) in (5, 20)
    goals = '\n'.join('- ' + ('Increase ' if o['direction'] == 1 else 'Decrease ') +
                      old.LABELS[o['property']] + '.' for o in objectives)
    choices = '\n'.join('Choice {}:\n<SMILES> {} </SMILES>'.format(i + 1, c['raw_smiles'])
                        for i, c in enumerate(candidates))
    return old.JUDGE_TEMPLATE.format(source_smiles=record['source_smiles'], goals=goals,
                                    n_candidates=len(candidates), choices=choices)


def new_seed(model, task, idx, phase, parent=None):
    assert phase in ('missing_children', 'terminal_vote', 'development')
    # Separate namespace from old phases, stable across shard/resume schedules.
    return int(digest([EXECUTION, model, task, idx, phase, parent])[:15], 16) % 2147483647


def contract():
    original, old_hash = old.validate_freeze()
    return dict(execution=EXECUTION, method_id=METHOD, model_assembly=original,
        old_contract_sha256=old_hash, inputs_per_model=3905, output_slots=20,
        initial_candidates=5, parent_keep=4, children_per_parent=5, judge_votes=5,
        judge_generation='Unchanged raw prompt, sampling, single EOS,128 cap; parser-only repair',
        parser='Explicit complete integer; repeated same accepted; conflicting/list/out-of-range rejected',
        cache='Identical model/source/parent candidate ID/prompt; recorded old15-row batches retained',
        missing_children='One new five-row batch per missing initial-parent ID; child cap160',
        seed='sha256(execution,model,task,input_idx,phase,parentID)[:15] modulo2147483647',
        K1='Same-configuration final LM-vote rank0', K20='All20 terminal children ranked by LM votes',
        invalid_policy='Retain every empty/invalid/duplicate slot; no refill; no retry on invalid votes',
        model_quality_gate=False, property_access_during_search=False,
        disclosure='Residual unparseable/model-limited judge outputs remain; log counts, ties and fallback',
        development='Synthetic molecules independent of test inputs; parser/controller and GPU context checks',
        probe_tasks=['BDP', 'BDPQ', 'BHMQ'], probe_indices=[0, 1, 2, 3],
        source_audit=str(BV / 'tot_c20_feasibility_20261014_v1/cache_feasibility_sources.json'))


def validate_freeze():
    spec = read(ROOT / 'contract.json')
    assert spec == contract(), 'Scientific contract changed'
    for path, expected in read(ROOT / 'freeze.json')['files'].items():
        assert sha(path) == expected, ('Frozen code/source changed', path)
    return spec, sha(ROOT / 'contract.json')


def old_evidence(model, record, sources):
    directory = OLD / 'runs' / ('tot_' + model + '_' + record['task_id']) / ('%04d' % record['input_idx'])
    values = {}
    for name in ('complete', 'initial_prefix', 'parent_vote', 'children'):
        path = directory / (name + '.json')
        assert sha(path) == sources[str(path)], ('Changed old result', str(path))
        values[name] = read(path)
    g, prefix, pv, cv = [values[name] for name in ('complete', 'initial_prefix', 'parent_vote', 'children')]
    assert prefix['record'] == record and g['prefix_digest'] == digest(prefix)
    assert g['model_id'] == MODEL_IDS[model] and prefix['initial_candidates'] == g['initial_candidates']
    objectives = old.task_spec(record['task_id'])['objectives']
    initial = g['initial_candidates']
    assert pv['request']['prompts'] == [judge_prompt(record, objectives, initial)] * 5
    assert g['parent_vote'] == old.vote_ranking(pv['outputs'], 5)
    parents = [initial[i] for i in g['selected_parent_indices']]
    assert cv['request']['prompts'] == [old.child_prompt(record, objectives, p) for p in parents for _ in range(5)]
    assert len(g['children']) == 15 and cv['request']['max_new_tokens'] == 160
    assert pv['request']['max_new_tokens'] == 128
    for call, phase in ((pv, 'parent_vote'), (cv, 'children')):
        assert call['request']['model'] == model and call['request']['contract_sha256'] == g['contract_sha256']
        assert call['request']['seed'] == old.batch_seed(model, record['task_id'], record['input_idx'], phase)
    families = {}
    for rank, parent in enumerate(parents):
        children = g['children'][rank * 5:(rank + 1) * 5]
        for branch, child in enumerate(children):
            row = rank * 5 + branch
            assert child['parent_candidate_id'] == parent['candidate_id']
            assert child['raw_text'] == cv['outputs'][row]['raw_text']
            assert child['raw_smiles'] == old.parse_smiles(child['raw_text'])['parsed_smiles']
            assert child['prompt_sha256'] == digest(cv['request']['prompts'][row])
            assert child['generation_seed'] == cv['request']['seed']
        families[parent['candidate_id']] = children
    return values, families, directory
