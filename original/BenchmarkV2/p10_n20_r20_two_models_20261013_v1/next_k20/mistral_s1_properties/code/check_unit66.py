"""Exercise only the pre-documented s1 QED exception through accepted CPU producers."""
from common import *
from evaluate import child

def main():
    require_allocation();verify_dependencies()
    source=ROOT/'BenchmarkV2/k20_remaining_three_v1/runs/s1_BDQ_0240_0320/s1_BDQ_0240_0320_b16/terminals/requests.json'
    original_hash=sha(source);old=read(source)
    rid='prop-request-v1-b57b9e826fb0bd4888ca229810622cf62d246a2e559db7777454938b1878e2bb'
    requests=[r for r in old['requests'] if r['request_id']==rid];assert len(requests)==1
    directory=OUT/'checks/unit66';inp=directory/'requests.json'
    write(inp,dict(task_id='BDQ',requests=requests,admet_columns=old['admet_columns']))
    admet=directory/'admet/admet_result.json';raw=directory/'standard/responses_raw.json'
    child('admet',inp,admet.parent,directory)
    child('standard',inp,raw.parent,directory,admet)
    result=read(raw);assert len(result)==1 and result[0]['request_id']==rid
    normalize=directory/'normalize_input.json.gz';write(normalize,dict(cache_contract=old['cache_contract'],responses=result))
    normalized=directory/'normalized.json.gz';child('normalize',normalize,normalized,directory)
    r=read(normalized)[0]
    assert r['known_molecular_property_failure'] and r['rdkit_valid'] and r['comparison_properties'] is None
    assert r['property_error']['policy_id']=='rdkit_remove_hs_property_error_isolation_v1'
    assert r['property_error']['category']=='rdkit_kekulization_after_initial_parse'
    assert all(x['valid'] for x in r['canonical_representations'])
    assert sha(source)==original_hash
    write(directory/'audit.json',dict(status='PASS_ACCEPTED_UNIT66_EXCEPTION_ISOLATED',request_id=rid,source_path=str(source),source_sha256=original_hash,properties_absent_with_explicit_known_exception=True,gpu_used=False,normalized_sha256=sha(normalized)))
    print(read(directory/'audit.json'),flush=True)

if __name__=='__main__':main()
