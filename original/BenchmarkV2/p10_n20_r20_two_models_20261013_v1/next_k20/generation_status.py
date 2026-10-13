"""Lightweight file-commit coverage; property evaluation is a separate audit."""
import datetime
import json
from pathlib import Path

P=Path(__file__).resolve().parent
OLD=P.parents[1]/'k20_remaining_three_v1'

def read(p):return json.loads(p.read_text())

def main():
    original=new=missing=0;missing_batches=[]
    for unit in read(OLD/'units.json'):
        if unit['method']!='s1':continue
        for start in range(unit['start'],unit['stop'],4):
            batch=unit['unit_id']+'_b%02d'%((start-unit['start'])//4)
            rel=Path('runs')/unit['unit_id']/batch/'generation.json.gz'
            n=min(4,unit['stop']-start)*20
            if (OLD/rel).exists():original+=n
            elif (P/'mistral_s1_resume'/rel).exists():new+=n
            else:missing+=n;missing_batches.append(batch)
    icl=0;iclmissing=0;units_complete=0
    stage=P/'llama_icl_sampled'
    for unit in read(stage/'units.json'):
        directory=stage/'runs'/unit['unit_id']
        units_complete+=(directory/'complete.json').exists()
        for batch in unit['batches']:
            n=len(batch['input_uids'])*20
            if (directory/(batch['batch_id']+'.json.gz')).exists():icl+=n
            else:iclmissing+=n
    result=dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                scope='Atomic generation-file presence and frozen expected lane counts; final merged scientific validation/property metrics separate.',
                s1_mistral=dict(original_saved_slots=original,new_saved_slots=new,total_saved_slots=original+new,
                    missing_generation_slots=missing,missing_generation_batches=missing_batches,
                    generation_array_job='4722865',property_evaluation_complete_not_asserted=True),
                icl_llama_sampled=dict(saved_slots=icl,missing_generation_slots=iclmissing,
                    complete_generation_units=units_complete,total_units=56,generation_array_job='4722924',
                    property_evaluation_complete_not_asserted=True),
                main_k1_results_modified=False,nine_method_k20_complete=False)
    assert original+new+missing==78100 and icl+iclmissing==78100
    target=P/'generation_status.json';temporary=P/'generation_status.json.tmp'
    temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(target)
    summary=dict(result,s1_mistral={k:v for k,v in result['s1_mistral'].items() if k!='missing_generation_batches'})
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
