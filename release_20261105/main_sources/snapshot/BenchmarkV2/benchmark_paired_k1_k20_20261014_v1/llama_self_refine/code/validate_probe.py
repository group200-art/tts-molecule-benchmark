"""CPU structural/timing gate after GPU engineering probe; never gate on molecular success."""
from common import *


def main():
    cfg = verify()
    for name in ('cpu_checks.json','tokenizer_check.json'):
        assert read(ROOT/'reports'/name)['status'].startswith('PASS')
    units = read(ROOT/'units.json')
    templates = read(cfg['templates_path'])
    timings, identities, count, response_count = [], [], 0, 0
    for index in cfg['probe_unit_indices']:
        unit = units[index]
        batch = unit['batches'][0]
        path = ROOT/'generation'/'units'/unit['unit_id']/(batch['batch_id']+'.json.gz')
        result = read(path)
        audit_batch(result,batch,templates)
        count += len(result['lanes'])
        for position,(role,r) in enumerate(ROLES):
            callpath = ROOT/'generation'/'calls'/batch['batch_id']/('{}_{}.json.gz'.format(role,r))
            call = read(callpath)
            assert call['status']=='complete'
            expected = [lane['feedbacks'][r] if role=='feedback' else lane['answers'][r] for lane in result['lanes']]
            assert call['outputs']==expected
            assert call['identity']==call_identity(batch,role,r,call['prompts'])
            assert len(call['outputs'])==len(ordered_lanes(batch))==76
            assert all(call['effective_decoding'][k]==v for k,v in cfg['decoding'].items())
            assert call['effective_decoding']['max_new_tokens']==cfg['role_caps'][role]
            assert call['effective_decoding']['eos_token_id']==128009
            assert all(call['input_ids'][i][0]==128000 and len(call['input_ids'][i])==out['input_tokens']
                       and len(out['generated_ids'])==out['output_tokens'] and out['output_tokens']<=cfg['role_caps'][role]
                       for i,out in enumerate(call['outputs']))
            assert result['measurements'][position]==call['measurement']
            assert call['measurement']['model_generate_seconds']>0
            response_count += len(call['outputs'])
            identities.append(dict(path=str(callpath),sha256=sha(callpath)))
        timings.append(dict(task=unit['task_id'],inputs=4,terminals=76,
            generate_seconds=sum(x['model_generate_seconds'] for x in result['measurements']),
            peak_allocated_bytes=max(x['peak_allocated_bytes'] for x in result['measurements'])))
        identities.append(dict(path=str(path),sha256=sha(path)))
    assert count==228 and response_count==912
    gate = dict(status='PASS',contract_sha256=sha(ROOT/'contract.json'),freeze_sha256=sha(ROOT/'reports/freeze.json'),
        inputs=12,new_terminals=228,responses=912,timings=timings,artifacts=identities,
        reused_in_production=True,property_predictions=0,quality_or_success_gating=False,
        caution='Generation gate only; full3905 completeness and CPU property evaluation remain outstanding.')
    save(ROOT/'reports/engineering_gate.json',gate)
    print(json.dumps(gate,ensure_ascii=False))


if __name__ == '__main__':
    main()
