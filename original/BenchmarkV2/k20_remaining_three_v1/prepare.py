"""Freeze all execution choices before any new stochastic outcomes are observed."""
import time
from common import *


def main():
    frozen = read(FROZEN / 'protocol.json')
    cfg = dict(execution=EXECUTION, created_unix=time.time(),
               authorization='User: 我同意, 请补齐, 大概需要多长时间; complete remaining three full3905 K20 methods',
               source_protocol=str(FROZEN / 'protocol.json'), source_protocol_sha256=sha(FROZEN / 'protocol.json'),
               extensions=frozen['extensions'], tasks=TASKS, methods=METHODS, k=20, inputs_per_method=3905,
               physical_batch='4 consecutive manifest inputs x20 trajectories; no length/quality sorting; last batch may be smaller',
               unit='80 consecutive inputs from one task; max two concurrent GPU workers',
               sampling='native HF categorical sampling T1 top_p1 top_k0 beam1 for every role',
               rng='seed = first15hex SHA256([execution,batch_id,ordered exact call request keys]) modulo2147483647; reset for each physical call',
               rng_scope='reproducible batched streams; not bitwise equivalent to serial seeds; all20 trajectories have separate rows and fresh controller state',
               padding='left pad exact original slow-tokenizer input IDs, pad EOS with attention-mask0; required for variable-length decoder-only batches',
               s1='unchanged cumulative512 incl EOS, at most one Wait/min1, final cap256; per-row stop preserves each remaining budget',
               reflexion='unchanged original controller; first success or3trials; binary/invalid feedback only; all20 independent memories executed',
               icl='same frozen input-specific4 demonstrations and cap100; no property feedback',
               properties='persistent CPU model, unchanged original algorithms/assets, raw values + original py36 round2 CSV normalization; abort infrastructure errors',
               method_fairness='same model, full3905 inputs,20 evaluated terminals and directional rule; no compute/tool-access matching claim',
               slots='preserve duplicates, invalid and empty slots; no refill; no quality-selected configuration or seeds',
               probe='first4 BDP, BDPQ, BHMQ inputs/method, all20 trajectories; directly reused in formal full dataset; engineering/timing gates only',
               max_concurrent_gpus=2, prior_results_observed=True, preregistration_claim=False)
    save(ROOT / 'plan.json', cfg)
    all_units = []
    probes = []
    for method in METHODS:
        for task in TASKS:
            rows = manifest(task)
            for start in range(0, len(rows), 80):
                stop = min(start + 80, len(rows))
                unit = dict(index=len(all_units), unit_id='{}_{}_{:04d}_{:04d}'.format(method, task, start, stop),
                            method=method, task_id=task, start=start, stop=stop, records_sha256=digest(rows[start:stop]))
                all_units.append(unit)
                if start == 0 and task in ('BDP', 'BDPQ', 'BHMQ'):
                    probes.append(unit['index'])
    assert len(all_units) == 168 and len(probes) == 9
    assert sum(u['stop'] - u['start'] for u in all_units) == 3905 * 3
    save(ROOT / 'units.json', all_units)
    save(ROOT / 'probe_units.json', probes)
    refresh_inventory()
    print(json.dumps(dict(units=len(all_units), probes=probes, inputs=3905, terminal_slots=234300)), flush=True)


def refresh_inventory():
    sources = list(ROOT.glob('*.py')) + [ROOT / 'plan.json', ROOT / 'units.json', ROOT / 'probe_units.json']
    sources += list((LEGACY / 'src').glob('*.py')) + list((LEGACY / 'configs').glob('*.json'))
    sources += [LEGACY / name for name in ['standard_producer.py', 'admet_worker.py', 'rdkit_recheck_worker.py',
                                         'reflexion_evaluator_source.py']]
    sources += [FROZEN / 'protocol.json', FROZEN / 'extension_dependencies.json',
                REPO / 'BenchmarkV2/src/metrics_cpu.py', REPO / 'BenchmarkV2/registries/task_registry.draft.json',
                REPO / 'prompter.py', REPO / 'templates/simple.json', REPO / 'config.py']
    sources += [Path(task_contract(t)['manifest_path']) for t in TASKS]
    spec = read(LEGACY / 'configs/properties_reflexion.json')
    sources += [Path(p) for p in spec['identity_files']]
    sources += [REPO / 'BenchmarkV2/advisory_deliverables/core_tables_revision_20260918_v1/table_data.json',
                REPO / 'BenchmarkV2/advisory_deliverables/core_tables_revision_20260918_v1/main_per_input_compact.jsonl.gz',
                REPO / 'BenchmarkV2/k20_nine_full_20261003_v1/p10_comparison/final_comparison.json',
                REPO / 'BenchmarkV2/self_refine_r1_k20_v1/results/summary.json']
    inventory = [dict(path=str(p), sha256=sha(p)) for p in sorted(set(sources))]
    save(ROOT / 'inventory.json', inventory, replace=(ROOT / 'inventory.json').exists())


if __name__ == '__main__':
    main()
