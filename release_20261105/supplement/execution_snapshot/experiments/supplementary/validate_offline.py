"""Audit derived selector endpoints against immutable original per-input labels."""
from common import *
require_allocation()
checks=[]
for b in MODELS:
 old={r['record']['input_uid']:r for r in lines(OUT/'original'/(b+'__voting.jsonl.gz'))}
 n=0;disagreements=0
 for task in TASKS:
  root=OUT/'vote_evaluation'/b/task;done=read(root/'completion.json');assert done['status']=='COMPLETE' and done['inputs']==COUNTS[task]
  for p in (root/'inputs').glob('*.json'):
   r=read(p);original=old[r['record']['input_uid']];vote=r['selectors']['Vote'];assert len(r['proposals'])==400
   for k in KS:assert vote['SR@'+str(k)]==float(any(c['success'] for c in original['candidates'][:k]))
   assert vote['validity']==sum(c['structural_valid'] for c in original['candidates'])/20
   for proposal in r['proposals']:
    if proposal['original_proposal']['rdkit_valid_ev']!=proposal['structural_valid']:disagreements+=1
   for k in KS:
    upper=r['selectors']['Diagnostic group success upper bound']['SR@'+str(k)]
    assert all(r['selectors'][s]['SR@'+str(k)]<=upper+1e-12 for s in ['Vote','First proposal per group','Uniform position per group'])
   assert r['selectors']['Diagnostic group success upper bound']['validity'] is None
   n+=1
 assert n==3905
 checks.append(dict(backbone=b,inputs=n,proposal_positions=n*400,original_vote_endpoints_matched=n*5,selector_metric_validity_disagreements_preserved=disagreements,status='PASS'))
write(OUT/'offline_validation.json',dict(status='PASS',cases=checks))
audit=read(OUT/'blocker_investigation.json')
resolution=dict(status='RESOLVED_WITHOUT_PROTOCOL_CHANGE',original_report_sha256=sha(OUT/'blocker_investigation.json'),candidate_exception_cause='All four affected raw candidates were already invalid under the unchanged authoritative inf parser. Only an unnecessary later property-support probe was skipped.',error_policy_extended=False,selector_assertion_cause='Original selector canonical-roundtrip validity is stricter than original metric raw-parse validity; both definitions retained.',checks=read(OUT/'guard_checks.json'),all_vote_tasks_complete=True,formal_submission_requires_completed_pilot=True)
write(OUT/'blocker_resolution.json',resolution);log('offline_validation_passed',inputs=7810,protocol_changed=False)
