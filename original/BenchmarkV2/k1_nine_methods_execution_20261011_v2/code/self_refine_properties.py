"""Reusable CPU terminal-property evaluation; no generation or candidate selection."""
import subprocess
import time
from self_refine_common import *
from property_bridge import requests_for


def child(role, data_path, output_path, directory, admet=None):
    name = role + '_' + str(os.environ['SLURM_JOB_ID'])
    attempt = check_output(Path(directory) / 'attempts' / name / 'scope').parent
    # A failed producer remains in its own attempt directory. Only a fully
    # successful output is published, so explicit resumed jobs need no deletion.
    staged_output = attempt / ('result.json.gz' if role == 'normalize' else 'producer_output')
    args = [ENVS['gellmo_inf' if role == 'admet' else 'gellmo_eval_py36'], '-B',
            str(ROOT / 'code/self_refine_property_worker.py'), role, '--input', str(data_path),
            '--input-sha', sha(data_path), '--output', str(staged_output)]
    if admet is not None:
        args.extend(['--admet', str(admet)])
    started = time.perf_counter()
    with (attempt / 'stdout').open('x') as out, (attempt / 'stderr').open('x') as err:
        result = subprocess.run(args, stdout=out, stderr=err, cwd=str(ROOT), env=isolated_env(attempt))
    write(attempt / 'process.json', dict(argv=args, returncode=result.returncode,
                                        wall_seconds=time.perf_counter() - started))
    assert result.returncode == 0, 'CPU property child failed: ' + str(attempt / 'stderr')
    assert not Path(output_path).exists(), 'Do not replace committed scientific outputs'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    os.rename(str(staged_output), str(output_path))


def evaluate_candidates(records, candidates, output_dir, model_id,
                        property_method='self_refine', cached_responses=None):
    """One task, <= arbitrary candidates; exact raw/source/contract cache identity.

    Output is a filled candidate list in original order. This API never modifies
    candidates, chooses a winner, accesses success flags, or launches generation.
    Call from a CPU Slurm allocation with CUDA_VISIBLE_DEVICES=''.
    """
    require_allocation()
    task, = {r['task_id'] for r in records}
    assert len({r['input_uid'] for r in records}) == len(records)
    recs = {r['input_uid']: r for r in records}
    assert all(c['input_uid'] in recs and c['source_smiles'] == recs[c['input_uid']]['source_smiles'] for c in candidates)
    output_dir = check_output(Path(output_dir) / 'scope').parent
    filename = 'properties_' + property_method + '.json' if property_method in ('self_refine', 'fewshot', 'reflexion') else 'properties.json'
    config = read(LLAMA / 'configs' / filename)['tasks'][task]
    requests, mapping = requests_for(candidates, config['cache_contract'])
    wanted = {r['request_id']: r for r in requests}
    cache, origins = {}, {}
    for entry in cached_responses or []:
        assert sha(entry['path']) == entry['sha256']
        for response in read(entry['path']):
            rid = response['request_id']
            if rid not in wanted:
                continue
            identity = wanted[rid]['identity']
            assert all(response[k] == identity[k] for k in ('raw_smiles', 'source_smiles', 'cache_contract'))
            if rid in cache:
                assert cache[rid] == response, 'Conflicting immutable property caches'
            cache[rid], origins[rid] = response, entry
    missing = [r for r in requests if r['request_id'] not in cache]
    write(output_dir / 'request_inventory.json', dict(task_id=task, model_id=model_id,
        candidate_count=len(candidates), unique_requests=len(requests), cache_hits=len(cache),
        new_requests=len(missing), reused_sources=origins, source_contract=config['cache_contract']))
    for start in range(0, len(missing), 100):
        directory = check_output(output_dir / ('block{:03d}'.format(start // 100)) / 'scope').parent
        batch = missing[start:start + 100]
        inp = directory / 'requests.json'
        write(inp, dict(task_id=task, requests=batch, admet_columns=config['admet_columns']))
        admet = directory / 'admet/admet_result.json'
        if not admet.exists():
            child('admet', inp, directory / 'admet', directory)
        responses_path = directory / 'standard/responses_raw.json'
        if not responses_path.exists():
            child('standard', inp, directory / 'standard', directory, admet)
        responses = read(responses_path)
        assert {r['request_id'] for r in responses} == {r['request_id'] for r in batch}
        cache.update({r['request_id']: r for r in responses})
        write(directory / 'complete.json', dict(status='complete', input_sha256=sha(inp),
                                                response_sha256=sha(responses_path)))
        print(json.dumps(dict(event='property_block_complete', task_id=task,
                              completed_new_requests=min(start + 100, len(missing)), total_new_requests=len(missing))), flush=True)
    payload = dict(task_id=task, candidates=candidates, requests=requests, mapping=mapping,
                   responses=[cache[r['request_id']] for r in requests], cache_contract=config['cache_contract'])
    inp = output_dir / 'normalize_input.json.gz'
    write(inp, payload)
    final = output_dir / 'filled_candidates.json.gz'
    if not final.exists():
        child('normalize', inp, final, output_dir)
    filled = read(final)
    assert [c['candidate_id'] for c in filled] == [c['candidate_id'] for c in candidates]
    write(output_dir / 'complete.json', dict(status='complete', candidate_count=len(filled),
        candidates_sha256=sha(final), input_sha256=sha(inp), known_molecular_failures=sum(
            c['known_molecular_property_failure'] for c in filled)))
    return filled
