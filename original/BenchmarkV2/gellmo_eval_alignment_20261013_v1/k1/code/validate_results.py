"""Independent lightweight consistency and immutable-input checks after CPU run."""
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

OUT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def csvrows(name):
    with (OUT/name).open() as f:return list(csv.DictReader(f))

def close(a,b):return math.isclose(float(a),float(b),rel_tol=1e-11,abs_tol=1e-10)

def main():
    audit=json.loads((OUT/'audit.json').read_text())
    assert audit['status']=='PASS' and audit['endpoints']==70290 and audit['task_groups']==180
    assert audit['statistics']['RI_legacy_cached_mismatches']==0
    assert audit['SR_mismatch_groups']==[]
    assert audit['new_generation']==audit['new_target_property_predictions']==audit['endpoint_changes']==0
    for name,digest in audit['output_sha256'].items():assert sha(OUT/name)==digest,name
    for path,digest in audit['provenance']['endpoint_source_hashes'].items():assert sha(path)==digest,path
    for name,item in audit['provenance']['snapshots'].items():assert sha(item['source_path'])==item['sha256']==sha(OUT/'sources'/name),name
    assert sha(OUT/'code/evaluate_k1.py')==audit['provenance']['script_sha256']
    contract=json.loads((OUT/'metric_contract.json').read_text())
    assert sha(contract['reference']['path'])==contract['reference']['sha256']
    assert sha(contract['reference']['canonical_snapshot_path'])==contract['reference']['canonical_snapshot_sha256']
    tasks=csvrows('task_metrics.csv');macro=csvrows('macro_metrics.csv');aps=csvrows('aps_by_task.csv')
    assert len(tasks)==180 and len(macro)==54 and len(aps)==630
    configs=defaultdict(list)
    for row in tasks:
        configs[row['model_id'],row['method_id']].append(row)
        assert close(row['SR'],100*int(row['n_success'])/int(row['n_inputs']))
        assert close(row['Val_legacy_rdkit'],100*int(row['n_valid_legacy_rdkit'])/int(row['n_inputs']))
        assert close(row['Val_current'],100*int(row['n_valid_current'])/int(row['n_inputs']))
        assert close(row['RI_official_code'],row['RI_legacy_cached'])
        assert int(row['n_success'])>0
    assert len(configs)==18
    for key,rows in configs.items():
        assert len(rows)==10 and sum(int(r['n_inputs']) for r in rows)==3905
        final=next(r for r in macro if (r['model_id'],r['method_id'])==key and r['scope']=='Macro10')
        for metric in ['SR','Val_legacy_rdkit','Val_current','RI_official_code','Sim_raw','SAS_raw','Nov_official']:
            assert close(final[metric],math.fsum(float(r[metric]) for r in rows)/10),(key,metric)
    old=json.loads((OUT/'sources/task_metrics.json').read_text())
    baseline={(r['model_id'],r['method_id'],r['task_id']):r['metrics'] for r in old}
    for row in tasks:
        reference=baseline[row['model_id'],row['method_id'],row['task_id']]
        assert close(row['SR'],reference['SR_dir']) and int(row['n_success'])==reference['n_success_dir']
        assert close(row['Val_current'],reference['Val_slot'])
    for row in aps:
        reference=baseline[row['model_id'],row['method_id'],row['task_id']]
        assert close(row['APS'],reference['APS'][row['property']])
    report=dict(status='PASS',checked_utc=datetime.now(timezone.utc).isoformat(),endpoints=70290,task_groups=180,configurations=18,APS_task_property_rows=len(aps),SR_identical_all_groups=True,current_validity_identical_all_groups=True,RI_official_matches_legacy_all_successful_endpoints=True,all_generated_output_hashes_verified=True,all_endpoint_source_hashes_unchanged=True,all_snapshot_originals_unchanged=True,quality_means_and_APS_verified=True,audit_sha256=sha(OUT/'audit.json'),validation_script_sha256=sha(__file__))
    (OUT/'final_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
