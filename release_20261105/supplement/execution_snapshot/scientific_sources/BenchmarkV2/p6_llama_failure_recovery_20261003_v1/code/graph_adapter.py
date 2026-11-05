"""Molecular task interfaces around byte-preserved author GoT operations."""
import json
import re
import sys
import time
from common import ROOT, CONTRACT, task_contract, text_sha
sys.path.insert(0, str(ROOT / 'vendor'))
from graph_of_thoughts.operations import Generate, Score, KeepBestN, Improve, Aggregate, Selector, GraphOfOperations
from graph_of_thoughts.controller import Controller
from graph_of_thoughts.prompter import Prompter
from graph_of_thoughts.parser import Parser
from graph_of_thoughts.language_models import AbstractLanguageModel
from scorer import score_pool

PAIRS = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
GROUPS = [(1, 2, 3, 4), (1, 2, 3, 5), (1, 2, 4, 6), (1, 3, 5, 6)]


def parse_answer(text):
    stripped = text.strip()
    tags = re.findall(r'<SMILES>\s*(.*?)\s*</SMILES>', stripped, flags=re.I | re.S)
    nonempty = [s.strip() for s in tags if s.strip()]
    lines = [s.strip() for s in stripped.splitlines() if s.strip()]
    return {'parsed_smiles': nonempty[-1] if nonempty else (lines[0] if lines else ''),
            'parse_method': 'last_nonempty_tag' if nonempty else ('first_nonempty_line' if lines else 'empty'),
            'tag_count': len(tags), 'nonempty_tag_count': len(nonempty),
            'parser_version': 'P8-LAST-NONEMPTY-TAG-FIRSTLINE-v1', 'raw_response': text}


def unused(*args, **kwargs):
    raise RuntimeError('Unapproved LLM scoring/validation operation')


class MolecularLM(AbstractLanguageModel):
    def __init__(self, context, backend):
        self.ctx, self.backend = context, backend
        self.prompt_tokens = self.completion_tokens = self.cost = 0

    def query(self, query, num_responses=1):
        spec = self.ctx.pending
        assert spec is not None and spec['prompt'] == query
        expected = self.ctx.G if spec['operation'] == 'Generate' else 1
        assert num_responses == expected
        return self.backend.query(query, num_responses, spec)

    def get_response_texts(self, query_responses):
        return query_responses


class MolecularPrompter(Prompter):
    def __init__(self, context):
        self.ctx = context

    def parent_block(self, state):
        c = self.ctx
        n = c.byid[state['node_id']]
        r = c.props[n['node_id']]
        score = c.score_maps['wave0'][n['node_id']]
        full = json.dumps(score['components'], ensure_ascii=False, allow_nan=False, sort_keys=False).strip()
        rendered = full if len(full) <= 800 else full[:800] + '...[truncated]'
        c.feedback[n['node_id']] = {'full': full, 'rendered': rendered, 'truncated': len(full) > 800,
                                    'full_chars': len(full), 'rendered_chars': len(rendered),
                                    'full_sha256': text_sha(full), 'rendered_sha256': text_sha(rendered)}
        values = dict(local_node_id=n['local_node_id'], wave0_rank=str(score['rank']),
                      rdkit_valid=str(r['rdkit_valid']), wave0_score=repr(score['value_score']),
                      parent_smiles=r['inf_canonical'] if r['rdkit_valid'] else (n['parsed_smiles'] or 'INVALID'),
                      rendered_score_components=rendered)
        return CONTRACT['prompt_contract']['field_definitions']['parent_block'].format(**values)

    def render(self, operation, parents, local_id):
        c = self.ctx
        fields = CONTRACT['prompt_contract']['field_definitions']
        values = {'source_smiles': c.record['source_smiles'].strip(),
                  'goal_lines': '\n'.join(fields['goal_lines'][o['property']] for o in c.task['objectives'])}
        if operation == 'Refine':
            values['parent_block'] = self.parent_block(parents[0])
            valid = c.props[parents[0]['node_id']]['rdkit_valid']
            values['validity_note'] = fields['validity_notes']['valid' if valid else 'invalid']
        elif operation in ('Merge', 'Aggregate'):
            values['parent_blocks'] = '\n\n'.join('Parent {}:\n{}'.format(i + 1, self.parent_block(p)) for i, p in enumerate(parents))
        prompt = CONTRACT['prompt_contract']['templates'][operation]['text'].format(**values).strip()
        c.pending = {'task_id': c.record['task_id'], 'input_uid': c.record['input_uid'],
                     'input_idx': c.record['input_idx'], 'G': c.G, 'r': 0,
                     'operation': operation, 'wave': 0 if operation == 'Generate' else 1,
                     'local_node_id': local_id, 'parents': [p['node_id'] for p in parents],
                     'prompt': prompt, 'source_context': True}
        return prompt

    def generate_prompt(self, num_branches, **kwargs):
        assert num_branches == 1 and not self.ctx.nodes
        return self.render('Generate', [], 'g00')

    def improve_prompt(self, **kwargs):
        i = self.ctx.refine_index
        assert kwargs['node_id'] == self.ctx.parents[i]
        self.ctx.refine_index += 1
        return self.render('Refine', [kwargs], 'r{:02d}'.format(i))

    def aggregation_prompt(self, state_dicts, **kwargs):
        i = self.ctx.aggregate_index
        self.ctx.aggregate_index += 1
        ranks = (PAIRS + GROUPS)[i]
        assert [s['node_id'] for s in state_dicts] == [self.ctx.parents[r - 1] for r in ranks]
        return self.render('Merge' if i < 6 else 'Aggregate', state_dicts,
                           ('m{:02d}'.format(i) if i < 6 else 'a{:02d}'.format(i - 6)))

    validation_prompt = score_prompt = unused


class MolecularParser(Parser):
    def __init__(self, context):
        self.ctx = context

    def states(self, texts):
        c = self.ctx
        spec = c.pending
        expected = c.G if spec['operation'] == 'Generate' else 1
        assert len(texts) == expected
        states = []
        for i, text in enumerate(texts):
            local = 'g{:02d}'.format(i) if spec['operation'] == 'Generate' else spec['local_node_id']
            node = {'node_id': c.prefix + '/' + local, 'local_node_id': local,
                    'method_id': c.method_id, 'task_id': c.record['task_id'], 'input_uid': c.record['input_uid'],
                    'G': c.G, 'r': 0, 'created_order': len(c.nodes), 'operation': spec['operation'],
                    'parents': list(spec['parents']), 'source_context': True,
                    'prompt_sha256': text_sha(spec['prompt'])}
            node.update(parse_answer(text))
            assert node['node_id'] not in c.byid
            c.nodes.append(node)
            c.byid[node['node_id']] = node
            # Complete immutable state whitelist: no scores or properties inherited from a parent.
            states.append(dict(node))
        c.pending = None
        c.persist('nodes_parsed')
        return states

    def parse_generate_answer(self, state, texts):
        return self.states(texts)

    def parse_improve_answer(self, state, texts):
        return self.states(texts)[0]

    def parse_aggregation_answer(self, states, texts):
        return self.states(texts)

    parse_validation_answer = parse_score_answer = unused


class MolecularGraph:
    def __init__(self, record, G, backend, properties, persist=lambda *x: None):
        assert G in (6, 12, 24)
        self.record, self.G, self.backend, self.property_service = record, G, backend, properties
        self.task = task_contract(record['task_id'])
        self.method_id = 'P8-G{}-N{}-K20-v1'.format(G, G + 16)
        self.prefix = '{}/{}/{}/G{}/r0'.format(self.method_id, record['task_id'], record['input_uid'], G)
        self.nodes, self.byid, self.props, self.scores, self.score_maps = [], {}, {}, {}, {}
        self.parents, self.feedback, self.pending = [], {}, None
        self.refine_index = self.aggregate_index = 0
        self.persist_callback = persist
        self.ran = False
        self.scoring_seconds = 0.0
        self.scoring_seconds_by_stage = {}
        self.execution_events = []
        self.property_service.begin_graph(record, G)
        self.build()

    def persist(self, event):
        self.persist_callback(event, self)

    def score(self, states, stage):
        expected = self.G if stage == 'wave0' else self.G + 16
        assert len(states) == expected == len(self.nodes)
        assert [s['node_id'] for s in states] == [n['node_id'] for n in self.nodes]
        self.property_service.fill(self.nodes, self.props, stage)
        t = time.perf_counter()
        result = score_pool(self.nodes, self.props, self.record['source_properties'], self.task['objectives'])
        elapsed = time.perf_counter() - t
        self.scoring_seconds += elapsed
        self.scoring_seconds_by_stage[stage] = elapsed
        self.scores[stage] = result
        self.score_maps[stage] = {s['node_id']: s for s in result['scores']}
        if stage == 'wave0':
            self.parents = result['ranked_node_ids'][:6]
        self.persist('scored_' + stage)
        values = [s['value_score'] for s in result['scores']]
        assert len(values) == len(states)
        return values

    def build(self):
        graph = GraphOfOperations()
        gen = Generate(1, self.G)
        score0 = Score(1, True, lambda states: self.score(states, 'wave0'))
        keep6, improve = KeepBestN(6, True), Improve()
        score0.add_predecessor(gen)
        keep6.add_predecessor(score0)
        improve.add_predecessor(keep6)
        selectors, aggregates = [], []
        for ranks in PAIRS + GROUPS:
            selector = Selector(lambda thoughts, ranks=ranks: [thoughts[r - 1] for r in ranks])
            selector.add_predecessor(keep6)
            aggregate = Aggregate(1)
            aggregate.add_predecessor(selector)
            selectors.append(selector)
            aggregates.append(aggregate)
        full = Score(1, True, lambda states: self.score(states, 'full'))
        for op in [gen, improve] + aggregates:
            full.add_predecessor(op)
        keep20 = KeepBestN(20, True)
        keep20.add_predecessor(full)
        ops = [gen, score0, keep6, improve] + selectors + aggregates + [full, keep20]
        roles = {gen: 'generate_wave0', score0: 'score_wave0', keep6: 'select_wave0', improve: 'refine6', full: 'score_full', keep20: 'select_full'}
        for i, (selector, aggregate) in enumerate(zip(selectors, aggregates)):
            roles[selector] = 'select_parent_subset_{:02d}'.format(i)
            roles[aggregate] = 'aggregate_branch_{:02d}'.format(i)
        for op in ops:
            graph.add_operation(op)
            assert len(op.predecessors) == len(set(op.predecessors))
            assert len(op.successors) == len(set(op.successors))
            original_execute = op.execute
            def measured_execute(*args, _op=op, _execute=original_execute, **kwargs):
                assert not any(e['operation_id'] == _op.id for e in self.execution_events)
                event = {'operation_id': _op.id, 'type': _op.operation_type.name, 'role': roles[_op], 'started': True, 'completed': False}
                self.execution_events.append(event)
                t = time.perf_counter()
                try:
                    result = _execute(*args, **kwargs)
                    event['completed'] = True
                    return result
                finally:
                    event['inclusive_wall_seconds'] = time.perf_counter() - t
            op.execute = measured_execute
        assert graph.roots == [gen] and graph.leaves == [keep20]
        self.graph, self.keep6, self.keep20 = graph, keep6, keep20
        self.generated_ops = [gen, improve] + aggregates
        self.controller = Controller(MolecularLM(self, self.backend), graph, MolecularPrompter(self), MolecularParser(self), {})

    def run(self):
        assert not self.ran
        self.ran = True
        self.controller.run()
        assert self.controller.run_executed and all(op.executed for op in self.graph.operations)
        assert len(self.nodes) == self.G + 16
        assert [t.state['node_id'] for t in self.keep6.get_thoughts()] == self.parents
        top20 = [t.state['node_id'] for t in self.keep20.get_thoughts()]
        assert top20 == self.scores['full']['ranked_node_ids'][:20]
        for op in self.generated_ops:
            for thought in op.get_thoughts():
                self.byid[thought.state['node_id']]['author_thought_id'] = thought.id
                self.byid[thought.state['node_id']]['author_operation_id'] = op.id
        self.persist('complete')
        return self.snapshot('complete')

    def snapshot(self, event):
        edges = []
        for n in self.nodes:
            if n['operation'] == 'Generate':
                edges.append({'from': 'source', 'to': n['node_id'], 'kind': 'lineage'})
            else:
                edges.extend({'from': p, 'to': n['node_id'], 'kind': 'lineage'} for p in n['parents'])
                edges.append({'from': 'source', 'to': n['node_id'], 'kind': 'source_context'})
        return {'event': event, 'record': self.record, 'G': self.G, 'method_id': self.method_id,
                'nodes': self.nodes, 'properties': self.props, 'scores': self.scores,
                'parents': self.parents, 'feedback': self.feedback, 'edges': edges,
                'scoring_seconds': self.scoring_seconds,
                'scoring_seconds_by_stage': self.scoring_seconds_by_stage,
                'execution_events': self.execution_events,
                'top20': self.scores.get('full', {}).get('ranked_node_ids', [])[:20],
                'control_graph': [{'id': op.id, 'type': op.operation_type.name, 'executed': op.executed,
                                   'predecessors': [p.id for p in op.predecessors], 'successors': [p.id for p in op.successors],
                                   'thought_ids': [t.id for t in op.get_thoughts()]} for op in self.graph.operations]}
