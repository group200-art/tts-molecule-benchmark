"""CPU checks for expensive production boundaries, without new property inference."""
import copy
import tempfile
import unittest
from common import *
from generate import Backend, execute_batch
from aggregate import aggregate_pool


class Checks(unittest.TestCase):
    def test_complete_fixed_sampling_manifest(self):
        frozen = read(ROOT / 'batch_manifest.json')
        seen = set()
        for seed in range(51004, 51021):
            for b in batches(seed):
                for rec in b['rows']:
                    key = (seed, rec['task_id'], rec['input_uid'])
                    self.assertNotIn(key, seen)
                    seen.add(key)
                expected, = [e for e in frozen if e['batch_id'] == b['batch_id']]
                self.assertEqual(expected['rows'], [[r['task_id'], r['input_uid']] for r in b['rows']])
                self.assertEqual([c['seed'] for c in expected['call_seeds']], [batch_seed(b, *r) for r in ROLES])
        self.assertEqual(len(seen), 66385)

    def test_r1_prompt_roles_and_terminal(self):
        b = next(batches(51004))
        b['rows'] = b['rows'][:2]
        b['batch_id'] = 'fixture_never_generated'
        b['row_identity_sha256'] = digest([(r['task_id'], r['input_uid']) for r in b['rows']])
        observed = []
        def fake(role, r, prompts, batch):
            observed.append((role, r))
            output = []
            for i, p in enumerate(prompts):
                text = 'Feedback on answer' if role == 'feedback' else '<SMILES>{}</SMILES>'.format('C' if r == 0 else 'CC')
                if role == 'refine':
                    self.assertIn('<SMILES>C</SMILES>', p)
                    self.assertIn('Feedback on answer', p)
                output.append(dict(raw_text=text, prompt_digest=digest(p), role=role, r=r,
                    seed=batch_seed(batch, role, r), batch_id=batch['batch_id'], batch_row=i))
            return output, dict(cuda_wall_seconds=0)
        result = execute_batch(b, fake, read(LEGACY / 'configs/prompts.json'))
        self.assertEqual(observed, ROLES)
        self.assertTrue(all(l['terminal_parse']['raw_smiles'] == 'CC' for l in result['lanes']))
        changed = copy.deepcopy(b['rows'][0])
        changed['source_properties'] = {k: 99999 for k in changed['source_properties']}
        self.assertEqual(task_text(changed), task_text(b['rows'][0]))

    def test_atomic_call_resume_checks_identity(self):
        b = next(batches(51004))
        prompts = ['fixed fixture']
        role, r = ROLES[0]
        ident = dict(batch_id=b['batch_id'], row_identity_sha256=b['row_identity_sha256'], role=role, r=r,
            seed=batch_seed(b, role, r), prompt_digests=[digest(x) for x in prompts], plan_sha256=sha(ROOT/'plan.json'))
        with tempfile.TemporaryDirectory(prefix='cpu-check-', dir=str(ROOT)) as tmp:
            d = Path(tmp)
            out = [dict(raw_text='already committed')]
            save(d/'calls'/b['batch_id']/'init_0.json.gz', dict(identity=ident, prompts=prompts,
                 outputs=out, measurement={}, complete=True))
            backend = Backend(plan(), d, d/'unused')
            self.assertEqual(backend.generate(role, r, prompts, b)[0], out)
            self.assertFalse(backend.loaded)
            with self.assertRaises(AssertionError):
                backend.generate(role, r, ['changed prompt'], b)

    def test_k20_any_success_keeps_invalid_and_duplicate_slots(self):
        rows = read(ROOT/'reuse/51001/BDP.json.gz')['rows']
        success = next(r for r in rows if r['metric_row']['SR_dir_success'])
        rec = next(r for r in manifest('BDP') if r['input_uid'] == success['input_uid'])
        base = success['candidate']
        pool = []
        for slot in range(20):
            c = copy.deepcopy(base)
            c.update(rep_seed=51001+slot, candidate_id='fixture-'+str(slot), raw_smiles='', rdkit_valid=False,
                     raw_properties=None, properties=None, property_row_exists=False)
            pool.append(c)
        self.assertFalse(aggregate_pool(rec, pool)['SR_dir_success'])
        pool[-1] = dict(base, rep_seed=51020, candidate_id='fixture-success')
        result = aggregate_pool(rec, pool)
        self.assertTrue(result['SR_dir_success'])
        self.assertEqual(result['selected_slot_id_dir'], 19)
        self.assertEqual(result['n_slots'], 20)
        duplicate_pool = [dict(base, rep_seed=51001+i, candidate_id='fixture-duplicate-'+str(i)) for i in range(20)]
        self.assertEqual(aggregate_pool(rec, duplicate_pool)['n_slots'], 20)
        with self.assertRaises(AssertionError):
            aggregate_pool(rec, pool[:19])
        with self.assertRaises(AssertionError):
            aggregate_pool(rec, list(reversed(pool)))
        pool[0]['original_slot_id'] = 4
        with self.assertRaises(AssertionError):
            aggregate_pool(rec, pool)


if __name__ == '__main__':
    verify()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.wasSuccessful()
    save(ROOT/'cpu_checks.json', dict(status='PASS', tests_run=result.testsRun,
         new_generations=0, new_property_predictions=0, plan_sha256=sha(ROOT/'plan.json')))
