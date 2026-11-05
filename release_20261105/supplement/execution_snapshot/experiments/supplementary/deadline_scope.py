"""Author-approved deadline scope; existing scientific generation code stays frozen."""
from common import *
import datetime

RELEASE = OUT.parent / 'recomb_deadline_freeze_20261104'
FREEZE_UTC = '2026-11-04T18:00:00+00:00'

def save(path, value):
 p=Path(path).resolve()
 assert RELEASE==p or RELEASE in p.parents, str(p)
 p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_name(p.name+'.'+str(os.getpid())+'.tmp')
 tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 os.replace(tmp,p)

def write_text(path, value):
 p=Path(path).resolve();assert RELEASE in p.parents
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value)

def csv_release(path,rows,fields=None):
 rows=list(rows);fields=fields or list(dict.fromkeys(k for r in rows for k in r))
 p=Path(path);assert RELEASE in p.resolve().parents;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:'NA' if v is None else json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()} for row in rows)

def now():return datetime.datetime.now(datetime.timezone.utc)

def unit_path(unit):
 return OUT/'generation'/'_'.join(str(unit[k]) for k in ['profile','backbone','seed','task','pair','start','stop'])

def prepare():
 RELEASE.mkdir(parents=True,exist_ok=True)
 rows=[r for r in lines(OUT/'subset_manifest.jsonl') if r['subset_rank']<50]
 assert len(rows)==500 and all(sum(r['task_id']==t for r in rows)==50 for t in TASKS)
 for r in rows:
  assert hashlib.sha256(('recomb-supp-v1|'+r['task_id']+'|'+r['input_uid']).encode()).hexdigest()==r['selection_hash']
 manifest=RELEASE/'subset_50_manifest.jsonl'
 write_text(manifest,''.join(json.dumps(r,separators=(',',':'),ensure_ascii=False)+'\n' for r in rows))
 units=[]
 for u in lines(OUT/'production_units.jsonl'):
  if u['seed']==1729 and u['pair']=='got' and u['start']<50:
   units.append(dict(u,index=len(units),original_unit_index=u['index']))
 assert len(units)==40 and sum(2*(u['stop']-u['start']) for u in units)==2000
 save(RELEASE/'units.json',units)
 cfg=dict(status='FROZEN_SCOPE_AWAITING_END_TO_END_GATE',created_utc=now().isoformat(),internal_freeze_utc=FREEZE_UTC,
  official_full_paper_deadline_utc='2026-11-05T11:59:00+00:00',
  authorization='User request: 50 inputs per task, 500 total, two backbones, GoT versus independent Sampling with the same scorer, one generation seed; launch only after a complete timing and validation gate.',
  inputs=500,inputs_per_task=50,backbones=MODELS,generation_seeds=[1729],arms=['GoT','Sampling with GoT scoring'],arm_input_runs=2000,
  selection='First 50 in each task of the original frozen SHA-256 ranking. Same inputs for both models and both arms; no outcome-based selection.',
  subset_sha256=sha(manifest),units_sha256=sha(RELEASE/'units.json'),parent_scientific_freeze_sha256=sha(OUT/'production_freeze.json'),
  scientific_generation_code_unchanged=True,legacy_raw_storage_profile='subset_100',
  storage_note='The immutable parent generator names raw chunks subset_100. This release includes only the independently frozen 500-input scope and never claims completion of the old 996-input, three-seed study.',
  partition='rtx6k_gpu',qos='gpu_rtx6k',max_concurrent_gpus=4,gpus_per_worker=1,cpus_per_worker=8,memory_gb_per_worker=64,
  runtime='Original torch/transformers/PEFT, identical Blackwell Server Edition GPU, BF16 base and FP32 adapters, unchanged batching and response-specific random streams.',
  end_to_end_pilot='Two hash-first inputs per task and model, GoT pair only, followed by scoring/log validation, full precision similarity, metrics, paired source intervals and export. Excluded from formal statistics.',
  formal_reuse='Only complete matching GoT paired chunks at seed 1729 with exact source identity and frozen code are reusable; all other saved outputs remain separate.',
  minimum_finish_reserve_seconds=10800,timing_safety_multiplier=1.25,
  offline_inputs=3905,offline_priorities=['Strict Tanimoto >0.6 on original returned prefixes K=1 and 20','GoT reordering of the same selected twenty candidates','Complete Vote fixed pools and property labels'],
  reflexion_ablation=False,three_seed_repeats=False,partial_scope_results_allowed=False,overleaf_push=False)
 save(RELEASE/'scope.json',cfg)
 for p,h in read(OUT/'production_freeze.json')['generation_code'].items():assert sha(p)==h
 save(RELEASE/'scope_preparation_check.json',dict(status='PASS',inputs=500,units=40,expected_arm_inputs=2000,frozen_code_unchanged=True))
 print(str(RELEASE),flush=True)

if __name__=='__main__':prepare()
