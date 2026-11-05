"""Meaningful CPU boundaries: no cross-lane feedback, fixed terminal, exact resume, and full coverage."""
import copy
import tempfile
import unittest
from common import *
from run import Backend


class Checks(unittest.TestCase):
    def test_full_coverage_and_exact_probe_reuse(self):
        cfg = read(ROOT/'contract.json')
        units = read(ROOT/'units.json')
        keys = [(rec['task_id'], rec['input_uid'], seed)
                for u in units for b in u['batches'] for rec, seed in ordered_lanes(b)]
        self.assertEqual(len(keys), 74195)
        self.assertEqual(len(set(keys)), 74195)
        for task in TASKS:
            reused = read(ROOT/'reuse'/(task+'.json.gz'))
            self.assertEqual([x['record'] for x in reused], manifest(task))
            self.assertTrue(all(x['rep_seed']==51001 and x['repeat']==0 for x in reused))
        probe = [units[i]['batches'][0] for i in cfg['probe_unit_indices']]
        self.assertEqual([b['records'][0]['task_id'] for b in probe], ['BDP','BDPQ','BHMQ'])
        self.assertEqual(sum(len(b['records']) for b in probe), 12)
        self.assertEqual(sum(len(ordered_lanes(b)) for b in probe), 228)
        for b in probe:
            self.assertEqual([r['input_idx'] for r in b['records']], [0,1,2,3])

    def test_feedback_isolation_fixed_terminal_and_failures_retained(self):
        batch = copy.deepcopy(read(ROOT/'units.json')[0]['batches'][0])
        batch['records'] = batch['records'][:1]
        batch['batch_id'] = 'cpu-fixture-never-generated'
        observed = []
        def fake(role, r, prompts, b):
            observed.append((role,r))
            outputs = []
            for row, ((record, seed), prompt) in enumerate(zip(ordered_lanes(b), prompts)):
                if role == 'refine':
                    self.assertIn('INITIAL_LANE_'+str(seed), prompt)
                    self.assertIn('FEEDBACK_LANE_'+str(seed), prompt)
                    self.assertNotIn('FEEDBACK_LANE_'+str(seed+1), prompt)
                text = ('FEEDBACK_LANE_'+str(seed) if role=='feedback' else
                        '<SMILES>INITIAL_LANE_'+str(seed)+'</SMILES>' if role=='init' else
                        '' if row==0 else '<SMILES>CC</SMILES><SMILES>CCC</SMILES>')
                outputs.append(dict(raw_text=text, role=role, r=r, rep_seed=seed,
                    batch_id=b['batch_id'], batch_row=row, seed=batch_seed(b,role,r),
                    prompt_digest=digest(prompt)))
            return outputs, dict(model_generate_seconds=0)
        result = execute_batch(batch, fake, read(LLAMA/'configs/p5_prompts.json'))
        self.assertEqual(observed, ROLES)
        self.assertEqual(len(result['lanes']), 19)
        self.assertEqual(result['lanes'][0]['terminal_parse']['raw_smiles'], '')
        self.assertTrue(all(x['terminal_parse']['raw_smiles']=='CC' for x in result['lanes'][1:]))
        self.assertTrue(all(len(x['feedbacks'])==2 for x in result['lanes']))
        self.assertEqual(result['inference_property_calls'], 0)
        changed = copy.deepcopy(batch['records'][0])
        changed['source_properties'] = {'bbbp':999999,'drd2':999999,'plogp':999999}
        self.assertEqual(task_text(changed), task_text(batch['records'][0]))

    def test_resume_requires_same_prompt_lane_and_contract(self):
        batch = read(ROOT/'units.json')[0]['batches'][0]
        prompts = ['fixed '+str(i) for i in range(len(ordered_lanes(batch)))]
        identity = call_identity(batch,'init',0,prompts)
        with tempfile.TemporaryDirectory(prefix='cpu-resume-',dir=str(ROOT)) as tmp:
            directory = Path(tmp)
            outputs = [dict(raw_text='cached') for _ in prompts]
            save(directory/'calls'/batch['batch_id']/'init_0.json.gz',
                 dict(status='complete',identity=identity,prompts=prompts,outputs=outputs,measurement={}))
            backend = Backend(read(ROOT/'contract.json'),directory,directory/'not_loaded')
            self.assertEqual(backend.generate('init',0,prompts,batch)[0], outputs)
            self.assertFalse(backend.loaded)
            with self.assertRaises(AssertionError):
                backend.generate('init',0,list(reversed(prompts)),batch)
            altered = copy.deepcopy(batch)
            altered['records'].reverse()
            with self.assertRaises(AssertionError):
                backend.generate('init',0,prompts,altered)


if __name__ == '__main__':
    verify()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    assert result.wasSuccessful()
    save(ROOT/'reports/cpu_checks.json',dict(status='PASS',tests_run=result.testsRun,
        contract_sha256=sha(ROOT/'contract.json'),freeze_sha256=sha(ROOT/'reports/freeze.json'),
        new_generations=0,new_property_predictions=0))
