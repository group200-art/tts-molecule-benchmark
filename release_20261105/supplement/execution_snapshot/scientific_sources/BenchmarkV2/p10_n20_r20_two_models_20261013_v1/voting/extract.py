"""Replay twenty exact votes from the twenty original physical proposal blocks.

CPU/std-library only. Read frozen classified proposals, preserve all identities,
reset vote counts independently per block, and never inspect property success
until after every winner is fixed. No generation or molecular property inference.
"""
import ast
import csv
import gzip
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[2]
B = REPO / 'BenchmarkV2'
M = B / 'method_review_P10_v1/execution_vote_ev_v1'
L = B / 'p6_llama_main_ten_tasks_v1'
LR = B / 'p6_llama_failure_recovery_20261003_v1'
OLD_GROUP = B / 'k20_nine_full_20261003_v1/p10_comparison'
TASKS = ['BDP', 'BDQ', 'BPQ', 'DPQ', 'BDPQ', 'MPQ', 'BDMQ', 'BHMQ', 'BMPQ', 'HMPQ']
COUNTS = dict(zip(TASKS, [500] * 7 + [118, 191, 96]))
SOURCES = {}
EXPECTED = {}


def track(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    value = dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
    assert str(path) not in SOURCES or SOURCES[str(path)] == value
    if str(path) in EXPECTED:
        assert EXPECTED[str(path)] == value['sha256'], str(path)
    SOURCES[str(path)] = value
    return data


def read(path):
    data = track(path)
    if str(path).endswith('.gz'):
        data = gzip.decompress(data)
    return json.loads(data)


def lines(path):
    data = track(path)
    if str(path).endswith('.gz'):
        data = gzip.decompress(data)
    return [json.loads(x) for x in data.splitlines() if x.strip()]


def save(name, obj):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def emit(name, rows):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with gzip.open(tmp, 'wt') as f:
        for row in rows:
            f.write(json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n')
    os.replace(tmp, path)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=len(rows))


def count_vote(rows):
    counts, first = Counter(), {}
    for r in rows:
        key = (r['class_type'], r['class_value'])
        counts[key] += 1
        first.setdefault(key, r)
    keys = sorted(counts, key=lambda k: (-counts[k], first[k]['global_slot'], k))
    winner = first[keys[0]]
    return winner, dict(top_votes=counts[keys[0]], tie_count=sum(v == counts[keys[0]] for v in counts.values()),
                        class_count=len(counts), valid_slots=sum(r['rdkit_valid_ev'] for r in rows),
                        invalid_slots=sum(not r['rdkit_valid_ev'] for r in rows),
                        empty_raw_slots=sum(not r['raw_response'] for r in rows),
                        runner_up_votes=counts[keys[1]] if len(keys) > 1 else 0)


def compact(winner, stats, source_path, round_idx=None):
    return dict(round_idx=round_idx, candidate_id=winner['candidate_id'], raw_smiles=winner['parsed_candidate'],
                original_slot_id=winner['global_slot'], original_global_slot=winner['global_slot'],
                original_block=winner['block'], original_block_slot=winner['block_slot'], seed=winner['seed'],
                raw_response=winner['raw_response'], class_type=winner['class_type'], class_value=winner['class_value'],
                canonical_smiles=winner['canonical_smiles'], rdkit_valid_ev=winner['rdkit_valid_ev'],
                failure_reason=winner['failure_reason'], identity_exception=winner.get('identity_exception'),
                rdkit_version=winner['rdkit_version'], group_start_slot=20*round_idx if round_idx is not None else 0,
                group_end_slot=20*(round_idx+1)-1 if round_idx is not None else 399,
                N_used=20 if round_idx is not None else 400,
                source_proposal_path=str(source_path), source_proposal_sha256=SOURCES[str(source_path)]['sha256'],
                original_call_path=winner.get('call_path'), original_tokens_path=winner.get('tokens_path'), **stats)


def main():
    started = time.time()
    track(Path(__file__))
    track(OUT.parent / 'protocol.json')
    for path in [M/'src/p10_generation.py', M/'src/p10_vote.py', M/'src/p10_parser.py', M/'src/contracts.py',
                 M/'configs/generation.json', M/'configs/runtime.json', L/'worker.py', L/'src/p10_vote.py',
                 L/'src/llama_generation.py', L/'configs/p10_original_generation.json', L/'configs/full.json']:
        track(path)
    # Execute original pure voting function (prefix N=20 and N=400) as an
    # independent check; group function above only changes the window boundaries.
    tree = ast.parse((M/'src/p10_vote.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'prefix_vote')
    module = ast.Module(body=[node], type_ignores=[])
    digest = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    scope = dict(Counter=Counter, BUDGETS=(1,20,100,400), digest=digest)
    exec(compile(ast.fix_missing_locations(module), 'original_prefix_vote', 'exec'), scope)
    original_vote = scope['prefix_vote']
    parser_tree = ast.parse((M/'src/p10_parser.py').read_text())
    normalize_node = next(n for n in parser_tree.body if isinstance(n, ast.FunctionDef) and n.name=='normalize_raw_text')
    normalize_scope = dict(re=re)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[normalize_node],type_ignores=[])), 'original_normalize_raw_text','exec'),normalize_scope)
    normalize = normalize_scope['normalize_raw_text']
    manifests = {t: lines(B/'manifests'/(t+'.test.seen.jsonl')) for t in TASKS}
    manifest_maps = {t: {r['input_uid']:r for r in rows} for t,rows in manifests.items()}
    assert all(len(manifests[t]) == COUNTS[t] for t in TASKS)
    k1 = {}
    for model in ['mistral','llama']:
        rs = lines(B/'k1_nine_methods_execution_20261011_v2/results'/('cached_'+model+'__ev.jsonl.gz'))
        k1[model] = {(r['task_id'],r['input_uid']):r for r in rs}
        assert len(k1[model]) == len(rs) == 3905
    old_group_rows = lines(OLD_GROUP/'final_per_input_comparison.jsonl.gz')
    old_group = {(r['task_id'],r['input_uid']):r for r in old_group_rows}
    assert len(old_group) == 3905
    old_summary = read(OLD_GROUP/'final_comparison.json')
    original_index = read(M/'reports/raw_and_science_index.json')
    EXPECTED.update({r['path']:r['sha256'] for r in original_index})
    del original_index
    llama_index = read(LR/'results/raw_evidence_index.json')
    llama_units = [u for u in llama_index if u['unit']['method_id']=='ev']
    assert sum(u['unit']['n_inputs'] for u in llama_units) == 3905
    for unit in llama_units:
        EXPECTED.update({r['path']:r['sha256'] for r in unit['verified_files']})
    outputs, diagnostics = [], []
    model_stats = {}

    def transform(model, record, proposals, source_path, saved_winner=None):
        task,uid=record['task_id'],record['input_uid']
        expected=manifest_maps[task][uid]
        for key in ['task_id','input_uid','input_idx','source_smiles','source_properties']:
            assert record[key] == expected[key]
        assert len(proposals)==400
        assert len({r['candidate_id'] for r in proposals}) == 400
        assert [r['global_slot'] for r in proposals] == list(range(400))
        assert [r['original_slot_id'] for r in proposals] == list(range(400))
        expected_model='gellmo_p6_'+model
        for slot,r in enumerate(proposals):
            assert r['model_id']==expected_model and r['task_id']==task and r['input_uid']==uid
            assert r['block']==slot//20 and r['block_slot']==slot%20
            assert r['seed']==(10101+slot//20)*100000+TASKS.index(task)*1000+record['input_idx']
            assert r['rdkit_version']=='2024.03.6'
            assert r['class_value_sha256']==hashlib.sha256(r['class_value'].encode()).hexdigest()
            if r['rdkit_valid_ev']:
                assert r['class_type']=='VALID_CANONICAL' and r['class_value']==r['canonical_smiles'] and r['class_value']
            else:
                assert r['class_type']=='INVALID_RAW' and r['class_value']==normalize(r['raw_response'])
            if model=='mistral':
                assert r['prompt_sha256']==proposals[0]['prompt_sha256']
        winners=[]
        for b in range(20):
            group=proposals[20*b:20*(b+1)]
            pathfield='call_path' if model=='llama' else 'tokens_path'
            assert len({r[pathfield] for r in group})==1
            win,stats=count_vote(group)
            winners.append(compact(win,stats,source_path,b))
        assert len({proposals[20*b]['call_path' if model=='llama' else 'tokens_path'] for b in range(20)})==20
        whole,stats=count_vote(proposals)
        original=original_vote(proposals,[20,400])
        assert winners[0]['candidate_id']==original[0]['representative']['candidate_id']
        assert whole['candidate_id']==original[1]['representative']['candidate_id']
        for field in ['top_votes','tie_count','class_count']:
            assert winners[0][field]==original[0][field] and stats[field]==original[1][field]
        if saved_winner:
            assert whole['candidate_id']==saved_winner['representative']['candidate_id']
            for field in ['top_votes','tie_count','class_count']:
                assert stats[field]==saved_winner[field]
        cached=k1[model][task,uid]
        assert cached['source_properties']==record['source_properties']
        assert whole['candidate_id']==cached['selected_candidate']['candidate_id']
        assert whole['parsed_candidate']==cached['selected_candidate']['raw_smiles']
        native=compact(whole,stats,source_path)
        native['cached_directional_success']=cached['evaluation']['SR_dir_success']
        if model=='mistral':
            old=old_group[task,uid]
            assert len(old['winners'])==20 and old['old400_success']==native['cached_directional_success']
            for win,previous in zip(winners,old['winners']):
                assert (win['round_idx'],win['candidate_id'],win['original_slot_id'],win['raw_smiles'],win['class_type'],win['class_value']) == (previous['block'],previous['candidate_id'],previous['original_global_slot'],previous['raw_smiles'],previous['class_type'],previous['class_value'])
                assert type(previous['success']) is bool
                win['cached_directional_success']=previous['success']
                win['cached_success_provenance']=dict(path=str(OLD_GROUP/'final_per_input_comparison.jsonl.gz'),basis=previous['outcome_basis'])
                if 'property_request_id' in previous:win['property_request_id']=previous['property_request_id']
            assert any(w['cached_directional_success'] for w in winners)==old['new20x20_K20_success']
        row=dict(schema_version='P10-N20-R20-GROUPED-v1',model_id=expected_model,method_id='exact_voting_N20_R20',
                 task_id=task,input_uid=uid,input_idx=record['input_idx'],source_smiles=record['source_smiles'],
                 source_properties=record['source_properties'],N=20,R=20,K_out=20,original_proposals=400,
                 grouping='original_physical_calls_block0_to19',winners=winners,
                 first_group_N20_K1=dict(winners[0]),native_N400_reference=native,
                 distinct_winner_identities=len({(w['class_type'],w['class_value']) for w in winners}),
                 source_proposal_path=str(source_path),source_proposal_sha256=SOURCES[str(source_path)]['sha256'])
        return row

    # Llama is first to unblock new-winner property evaluation as early as possible.
    for model in ['llama','mistral']:
        model_rows=[]
        for task in TASKS:
            rows=[]
            if model=='llama':
                for u in [u for u in llama_units if u['unit']['task_id']==task]:
                    root=Path(u['run_directory'])
                    status=read(root/'status.json');assert status['status']=='complete'
                    path=root/'vote_result.json';v=read(path);assert v['property_access'] is False
                    assert len(v['inputs'])==u['unit']['n_inputs']
                    for item in v['inputs']:
                        rows.append(transform(model,item['record'],item['classified_proposals'],path,item['winner']))
            else:
                for phase in ['pilot','full']:
                    root=M/'cpu'/phase/task
                    records=read(root/'records.json')
                    for record in records:
                        path=root/'inputs'/('{:04d}'.format(record['input_idx']))/'classified.json.gz'
                        rows.append(transform(model,record,read(path),path))
            assert len(rows)==len({r['input_uid'] for r in rows})==COUNTS[task]
            rows.sort(key=lambda r:r['input_idx'])
            assert [r['input_uid'] for r in rows]==[r['input_uid'] for r in manifests[task]]
            outputs.append(emit('by_task/'+model+'__'+task+'.jsonl.gz',rows))
            save('task_readiness/'+model+'__'+task+'.json',dict(status='PASS_GROUPED_WINNERS',model=model,task=task,inputs=len(rows),terminal_slots=len(rows)*20,output=outputs[-1]))
            model_rows.extend(rows)
            wins=[w for row in rows for w in row['winners']]
            diag=dict(model_id='gellmo_p6_'+model,task_id=task,inputs=len(rows),groups=len(wins),
                      tied_max_groups=sum(w['tie_count']>1 for w in wins),all_unique_groups=sum(w['top_votes']==1 for w in wins),
                      all_same_groups=sum(w['top_votes']==20 for w in wins),invalid_winner_groups=sum(not w['rdkit_valid_ev'] for w in wins),
                      first20_tied_max=sum(r['winners'][0]['tie_count']>1 for r in rows),
                      mean_distinct_winner_identities=sum(r['distinct_winner_identities'] for r in rows)/len(rows),
                      group_top_vote_histogram=dict(sorted(Counter(w['top_votes'] for w in wins).items())),
                      group_tie_count_histogram=dict(sorted(Counter(w['tie_count'] for w in wins).items())))
            if model=='mistral':
                diag.update(N20_K1_success=sum(r['winners'][0]['cached_directional_success'] for r in rows),
                            N20_R20_K20_success=sum(any(w['cached_directional_success'] for w in r['winners']) for r in rows),
                            N400_K1_success=sum(r['native_N400_reference']['cached_directional_success'] for r in rows))
            diagnostics.append(diag)
            print(json.dumps(dict(model=model,task=task,inputs=len(rows),groups=len(wins),tied_max=diag['tied_max_groups'],all_unique=diag['all_unique_groups'])),flush=True)
        assert len(model_rows)==3905
        outputs.append(emit('grouped_inputs_'+model+'.jsonl.gz',model_rows))
        own=[d for d in diagnostics if d['model_id']=='gellmo_p6_'+model]
        stats={key:sum(r[key] for r in own) for key in ['inputs','groups','tied_max_groups','all_unique_groups','all_same_groups','invalid_winner_groups']}
        stats['model_id']='gellmo_p6_'+model
        stats['mean_distinct_winner_identities']=sum(r['distinct_winner_identities'] for r in model_rows)/3905
        stats['all20_winners_same_identity_inputs']=sum(r['distinct_winner_identities']==1 for r in model_rows)
        if model=='mistral':
            for newkey,oldkey in [('N20_K1','first20_K1_SR'),('N20_R20_K20','new20x20_K20_SR'),('N400_K1','old400_K1_SR')]:
                value=sum(100*d[newkey+'_success']/d['inputs'] for d in own)/10
                assert abs(value-old_summary['macro10'][oldkey])<1e-10
                stats[newkey+'_Macro10_SR']=value
        model_stats[model]=stats
        save('completion_'+model+'.json',dict(status='PASS',**stats,output=outputs[-1]))
    save('tie_diagnostics.json',dict(by_task=diagnostics,by_model=model_stats))
    save('source_provenance.json',SOURCES)
    save('output_manifest.json',outputs)
    save('audit.json',dict(status='PASS_FULL_TWO_MODEL_N20_R20_WINNERS',generated_utc=datetime.now(timezone.utc).isoformat(),
                          full_inputs_per_model=3905,terminal_slots_per_model=78100,total_terminal_slots=156200,
                          classified_proposals_examined=3124000,groups_examined=156200,
                          same_original_classifier=True,original_frozen_proposal_hashes_verified=True,
                          all_global_slot_block_seed_identities_verified=True,all_groups_are_disjoint_original_calls=True,
                          independent_counter_per_group=True,invalid_and_empty_votes_retained=True,
                          tie_rule='earliest original slot of tied class; earliest member of winning class',
                          original_N400_winner_identity_preserved=True,first_N20_prefix_original_function_identity=True,
                          mistral_all20_winners_match_complete_previous_grouped_evaluation=True,
                          mistral_three_reference_SR_macros_match=True,no_success_based_group_or_seed_selection=True,
                          no_new_LLM_generation=True,no_new_property_prediction=True,
                          inference_identity_environment='RDKit2024.03.6 classified proposals, unchanged',
                          scope='New operating point N20,R20,Kout20; original N400,R1,Kout1 retained separately. Not twenty N400 repeats.',
                          results=model_stats,elapsed_seconds=time.time()-started))
    print('PASS_FULL_TWO_MODEL_N20_R20_WINNERS',flush=True)


if __name__=='__main__':
    main()
