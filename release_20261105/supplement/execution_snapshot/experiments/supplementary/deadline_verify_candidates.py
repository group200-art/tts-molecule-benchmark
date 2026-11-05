"""Run the portable candidate reproduction on a complete pilot or formal scope."""
from deadline_scope import *
import argparse,subprocess

def main():
 require_allocation();p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');args=p.parse_args()
 target=RELEASE/('pilot_pipeline' if args.pilot else 'generation');count=20 if args.pilot else 500
 assert read(target/'status.json')['status']=='PASS_COMPLETE_FIXED_SCOPE'
 records=list(lines(OUT/'pilot_manifest.jsonl')) if args.pilot else list(lines(RELEASE/'subset_50_manifest.jsonl'))
 assert len(records)==count
 index=target/'input_manifest.csv';csv_release(index,[dict(task=r['task_id'],instance_id=r['input_uid'],raw_source_id=r['source_molecule_uid']) for r in records])
 identity=dict(pairs_sha256=sha(target/'paired_records.jsonl.gz'),metrics_sha256=sha(target/'per_input_metrics.csv'),manifest_sha256=sha(index),checker_sha256=sha(CODE/'reproduce_got_candidates.py'),scientific_source_manifest_sha256=sha(RELEASE/'release_bundle/evidence/scientific_source_manifest.json'),bridge_sha256=sha(RELEASE/'release_bundle/execution_snapshot/experiments/supplementary/bridges.py'))
 prior=read(target/'portable_candidate_check.json') if (target/'portable_candidate_check.json').exists() else None
 if prior and prior['status']=='PASS' and prior.get('cache_identity')==identity:
  print('Candidate reproduction already passed; all data and source hashes unchanged',flush=True);return
 subprocess.run([GEN_PY,'-B',str(CODE/'reproduce_got_candidates.py'),'--bundle',str(RELEASE/'release_bundle'),'--pairs',str(target/'paired_records.jsonl.gz'),'--metrics',str(target/'per_input_metrics.csv'),'--manifest',str(index),'--expected-count',str(count),'--output',str(target/'portable_candidate_check.json')],check=True)
 report=read(target/'portable_candidate_check.json');report['cache_identity']=identity;save(target/'portable_candidate_check.json',report)

if __name__=='__main__':main()
