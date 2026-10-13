"""Mechanical gate only. Success rates never decide whether full may run."""
import hashlib
import json
import zlib
import csv
import io
from common import ROOT, CONTRACT, read, unit, unit_key, unit_inputs, write, verify_execution, text_sha, utc
from graph_adapter import PAIRS, GROUPS, parse_answer
from scorer import score_pool


def jsonlines(path):
    with path.open() as f:
        for line in f:
            yield json.loads(line)


def expected_prompt(graph, node):
    pc = CONTRACT['prompt_contract']
    fields = pc['field_definitions']
    record = graph['record']
    task = next(t for t in CONTRACT['tasks'] if t['task_id'] == record['task_id'])
    values = {'source_smiles': record['source_smiles'].strip(),
              'goal_lines': '\n'.join(fields['goal_lines'][o['property']] for o in task['objectives'])}
    byid = {n['node_id']: n for n in graph['nodes']}
    scores = {s['node_id']: s for s in graph['scores']['wave0']['scores']}
    blocks = []
    for nid in node['parents']:
        n, r, s = byid[nid], graph['properties'][nid], scores[nid]
        full = json.dumps(s['components'], ensure_ascii=False, allow_nan=False, sort_keys=False).strip()
        rendered = full[:800] + '...[truncated]' if len(full) > 800 else full
        feedback = graph['feedback'][nid]
        assert feedback['full'] == full and feedback['rendered'] == rendered
        assert feedback['full_sha256'] == text_sha(full) and feedback['rendered_sha256'] == text_sha(rendered)
        blocks.append(fields['parent_block'].format(local_node_id=n['local_node_id'], wave0_rank=str(s['rank']),
                       rdkit_valid=str(r['rdkit_valid']), wave0_score=repr(s['value_score']),
                       parent_smiles=r['inf_canonical'] if r['rdkit_valid'] else (n['parsed_smiles'] or 'INVALID'),
                       rendered_score_components=rendered))
    op = node['operation']
    if op == 'Refine':
        values.update(parent_block=blocks[0], validity_note=fields['validity_notes']['valid' if graph['properties'][node['parents'][0]]['rdkit_valid'] else 'invalid'])
    elif op in ('Merge', 'Aggregate'):
        values['parent_blocks'] = '\n\n'.join('Parent {}:\n{}'.format(i + 1, b) for i, b in enumerate(blocks))
    return pc['templates'][op]['text'].format(**values).strip()


def validate_graph(graph):
    assert graph['event'] == 'complete'
    G = graph['G']
    nodes, props = graph['nodes'], graph['properties']
    assert len(nodes) == G + 16 and len(props) == G + 16
    assert [n['created_order'] for n in nodes] == list(range(G + 16))
    local_ids = ['g{:02d}'.format(i) for i in range(G)] + ['r{:02d}'.format(i) for i in range(6)] + ['m{:02d}'.format(i) for i in range(6)] + ['a{:02d}'.format(i) for i in range(4)]
    assert [n['local_node_id'] for n in nodes] == local_ids
    assert len({n['node_id'] for n in nodes}) == G + 16
    task = next(t for t in CONTRACT['tasks'] if t['task_id'] == graph['record']['task_id'])
    for stage, pool in [('wave0', nodes[:G]), ('full', nodes)]:
        assert score_pool(pool, props, graph['record']['source_properties'], task['objectives']) == graph['scores'][stage]
    assert graph['parents'] == graph['scores']['wave0']['ranked_node_ids'][:6]
    assert graph['top20'] == graph['scores']['full']['ranked_node_ids'][:20]
    assert len(graph['edges']) == G + 50
    assert sum(e['kind'] == 'lineage' for e in graph['edges']) == G + 34
    expected_edges = []
    ranks = [(j,) for j in range(1, 7)] + PAIRS + GROUPS
    for i, n in enumerate(nodes):
        assert n['parents'] == ([] if i < G else [graph['parents'][j - 1] for j in ranks[i - G]])
        parsed = parse_answer(n['raw_response'])
        assert all(n[k] == v for k, v in parsed.items())
        r = props[n['node_id']]
        assert r['request_smiles'] == (r['inf_canonical'] if r['rdkit_valid'] else n['parsed_smiles'])
        assert r['identity']['request_smiles'] == r['request_smiles']
        assert r['identity']['source_smiles'] == graph['record']['source_smiles']
        assert r['identity']['cache_contract']['preprocessing'] == 'P8-inf-canonical-request-v1'
        assert not r['environment_validity_disagreement']
        if r['rdkit_valid']:
            assert r['eval_valid'] and r['rounding_versions'] == {'pandas': '1.0.5', 'numpy': '1.18.5'}
            assert r['rounding_csv']
            table = list(csv.DictReader(io.StringIO(r['rounding_csv'])))
            assert len(table) == 1 and set(table[0]) == set(r['properties'])
            assert {p: float(v) if v else None for p, v in table[0].items()} == r['properties']
        assert isinstance(n['author_thought_id'], int) and isinstance(n['author_operation_id'], int)
        assert text_sha(expected_prompt(graph, n)) == n['prompt_sha256']
        if i < G:
            expected_edges.append({'from': 'source', 'to': n['node_id'], 'kind': 'lineage'})
        else:
            expected_edges.extend({'from': p, 'to': n['node_id'], 'kind': 'lineage'} for p in n['parents'])
            expected_edges.append({'from': 'source', 'to': n['node_id'], 'kind': 'source_context'})
    assert graph['edges'] == expected_edges
    for op in graph['control_graph']:
        assert op['executed'] and len(op['successors']) == len(set(op['successors']))
        assert len(op['predecessors']) == len(set(op['predecessors']))
    assert len(graph['execution_events']) == len(graph['control_graph']) == 26
    assert len({e['operation_id'] for e in graph['execution_events']}) == 26
    assert all(e['completed'] for e in graph['execution_events'])
    return True


def validate_unit(u, require_complete=True, verify_rng=True):
    directory = ROOT / 'workers' / unit_key(u)
    status = read(directory / 'unit_status.json')
    assert status['unit'] == u
    if require_complete:
        assert status['status'] == 'complete' and status['exception'] is None and not status['close_errors']
    records = unit_inputs(u)
    assert read(directory / 'inputs.json') == records
    expected = [(G, r['input_uid']) for G in [12, 6, 24] for r in records]
    done = [(g['G'], g['input_uid']) for g in status['completed_graphs']]
    assert done == expected[:len(done)] and len(done) == len(set(done))
    if require_complete:
        assert done == expected
    graphs = {}
    for entry in status['completed_graphs']:
        graph = read(ROOT / entry['graph'])
        validate_graph(graph)
        assert graph['record'] == next(r for r in records if r['input_uid'] == entry['input_uid'])
        graphs[(entry['G'], entry['input_uid'])] = graph
        from summarize import output_candidates
        from metrics import evaluate_registered
        for K in (1, 5, 10, 20):
            result = evaluate_registered(u['task_id'], entry['input_uid'], graph['record']['source_properties'], output_candidates(graph, K), K)
            assert result['n_slots'] == K and result['n_eligible_slots'] <= result['n_valid_slots'] <= K
    starts, results, last_rng = {}, {}, {}
    verified = set()
    binary = (directory / 'rng_states.bin').open('rb') if verify_rng else None
    try:
        initializations = list(jsonlines(directory / 'rng_initializations.jsonl'))
        for v in initializations:
            assert v['seed'] == u['wave{}_seed'.format(v['wave'])]
            last_rng[(v['G'], v['wave'])] = v['state']['sha256']
        for call in jsonlines(directory / 'calls.jsonl'):
            if call['event'] == 'precall_failure':
                if require_complete:
                    raise AssertionError('Precall failure in complete unit')
                continue
            index = call['call_index']
            if call['event'] == 'started':
                assert index not in starts
                starts[index] = call
                continue
            assert index in starts and index not in results
            assert call['started'] and (call['completed'] or not require_complete)
            results[index] = call
            c = call['context']
            assert c['task_id'] == u['task_id'] and c['G'] in (6, 12, 24)
            cap = 200 if c['wave'] == 0 else 220
            cfg = call['generation_config_passed']
            assert cfg['max_new_tokens'] == cap and cfg['top_k'] == 0 and cfg['num_return_sequences'] == 1
            assert cfg['do_sample'] is True and cfg['temperature'] == cfg['top_p'] == 1.0
            assert 1 <= len(call['input_ids']) <= 4096 and len(call['input_ids']) == call['prompt_tokens']
            assert call['input_ids'][0] == 1 and set(call['attention_mask']) == {1}
            assert text_sha(call['prompt']) == call['prompt_sha256']
            stream = (c['G'], c['wave'])
            assert call['rng_before']['sha256'] == last_rng[stream]
            if 'rng_after' in call:
                last_rng[stream] = call['rng_after']['sha256']
            elif require_complete:
                raise AssertionError('Missing post-call RNG')
            for key in ('rng_before', 'rng_after'):
                ref = call.get(key)
                if binary is not None and ref and ref['sha256'] not in verified:
                    binary.seek(ref['offset'])
                    data = zlib.decompress(binary.read(ref['compressed_bytes']))
                    assert hashlib.sha256(data).hexdigest() == ref['sha256']
                    verified.add(ref['sha256'])
            if call['completed']:
                ids = call['continuation_ids']
                assert 1 <= len(ids) <= cap
                assert call['tokens_through_first_eos'] == (ids.index(2) + 1 if 2 in ids else len(ids))
                graph = graphs.get((c['G'], c['input_uid']))
                if graph:
                    node = next(n for n in graph['nodes'] if n['local_node_id'] == c['local_node_id'])
                    assert node['raw_response'] == call['decoded_continuation_untrimmed']
                    assert node['prompt_sha256'] == call['prompt_sha256'] and node['parents'] == c['parents']
        if require_complete:
            assert len(starts) == len(results) == u['planned_generate_calls']
            assert list(starts) == list(range(u['planned_generate_calls']))
            expected_calls = [(G, r['input_uid'], local) for G in [12, 6, 24] for r in records
                              for local in (['g{:02d}'.format(i) for i in range(G)] + ['r{:02d}'.format(i) for i in range(6)] + ['m{:02d}'.format(i) for i in range(6)] + ['a{:02d}'.format(i) for i in range(4)])]
            assert [(c['context']['G'], c['context']['input_uid'], c['context']['local_node_id']) for c in results.values()] == expected_calls
            assert all(c['completed'] and c['exception'] is None for c in results.values())
            assert read(directory / 'generation_cost.json')['model_load_completed'] == 1
            identities = read(directory / 'model_identity.json')
            assert identities['tokenizer']['is_fast'] is False
            assert identities['adapter_status']['active_adapters'] == ['default'] and identities['adapter_status']['merged_adapters'] == []
            if u['task_id'] == 'DPQ':
                cost = read(directory / 'property_cost.json')['admet_shared_and_nested_cost']
                assert cost['model_load_calls'] == cost['predict_started'] == 0 and not cost['admet_imported']
            # Same UID initial prompt bytes across every physical sample and G.
            for r in records:
                prompts = {c['prompt'] for c in results.values() if c['context']['input_uid'] == r['input_uid'] and c['context']['wave'] == 0}
                assert len(prompts) == 1
    finally:
        if binary:
            binary.close()
    return {'unit_key': unit_key(u), 'completed_graphs': len(graphs), 'calls_started': len(starts),
            'calls_with_result': len(results), 'rng_states_verified': len(verified), 'mechanical_pass': True}


def pilot_audit():
    import traceback
    verify_execution()
    result = {'time_utc': utc(), 'gate_basis': 'mechanical_only_no_SR_threshold', 'pass': False, 'units': []}
    try:
        for index in (0, 1):
            result['units'].append(validate_unit(unit('pilot', index)))
        result['pass'] = True
    except BaseException as err:
        result.update(exception=repr(err), traceback=traceback.format_exc())
    write(ROOT / 'control/pilot_audit.json', result)
    return result
