"""Supplementary experiment boundaries, immutable source reads and manifests."""
import ast, csv, gzip, hashlib, importlib.util, json, math, os, sys, time
from collections import defaultdict
from pathlib import Path
sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parent
REPO = CODE.parents[1]
B = REPO / 'BenchmarkV2'
OUT = REPO / 'results/supplementary/recomb_supp_v1_20261103'
G = B / 'p6_llama_failure_recovery_20261003_v1'
M = B / 'method_review_P8_v1/design_after_p9_v1/execution_v1'
R = B / 'benchmark_completion_20261023_v1'
V = B / 'p10_n20_r20_two_models_20261013_v1'
TASKS = ['BDP','BDQ','BPQ','DPQ','BDPQ','MPQ','BDMQ','BHMQ','BMPQ','HMPQ']
COUNTS = dict(zip(TASKS, [500]*7+[118,191,96]))
METHODS = ['beam','sampling','self_refine','tot','got','s1','voting','icl','reflexion']
MODELS = ['mistral','llama']
SEEDS = [1729,2718,3141]
KS = [1,5,10,20]
DELTAS = dict(bbbp=.2,drd2=.2,plogp=1.,qed=.1,hia=.1,mutagenicity=.1)
GEN_PY = '/work/users/w/a/wanzhou/conda_envs/gellmo_mistral_gen/bin/python'
INF_PY = '/work/users/w/a/wanzhou/conda_envs/gellmo_inf/bin/python'
EVAL_PY = '/work/users/w/a/wanzhou/conda_envs/gellmo_eval_py36/bin/python'
SOURCES = {}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def track(path):
 p=Path(path).resolve()
 if str(p) not in SOURCES:SOURCES[str(p)]=dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
 return p
def read(path):
 p=track(path)
 with gzip.open(p,'rt') if p.suffix=='.gz' else p.open() as f:return json.load(f)
def lines(path, tracked=True):
 p=track(path) if tracked else Path(path)
 with gzip.open(p,'rt') if p.suffix=='.gz' else p.open() as f:
  for l in f:
   if l.strip():yield json.loads(l)
def safe(path):
 p=Path(path).resolve()
 if OUT not in p.parents and CODE not in p.parents:raise PermissionError('Output outside independent supplementary directories: '+str(p))
 p.parent.mkdir(parents=True,exist_ok=True);return p
def write(path,obj):
 p=safe(path);tmp=p.with_name(p.name+'.'+str(os.getpid())+'.tmp')
 tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n');os.replace(str(tmp),str(p))
def emit(path,rows):
 p=safe(path);tmp=p.with_name(p.name+'.'+str(os.getpid())+'.tmp');n=0
 with gzip.open(tmp,'wt') if p.suffix=='.gz' else tmp.open('w') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n');n+=1
 os.replace(str(tmp),str(p));return n
def append(path,obj):
 with safe(path).open('a') as f:f.write(json.dumps(obj,ensure_ascii=False,allow_nan=False)+'\n')
def csvout(path,rows,fields=None):
 rows=list(rows);fields=fields or list(dict.fromkeys(k for r in rows for k in r))
 p=safe(path);tmp=p.with_name(p.name+'.'+str(os.getpid())+'.tmp')
 with tmp.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:('NA' if v is None else json.dumps(v) if isinstance(v,(list,dict)) else v) for k,v in r.items()} for r in rows)
 os.replace(str(tmp),str(p))
def log(event,**kwargs):
 row=dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),event=event,**kwargs)
 append(OUT/'run.log',row);print(json.dumps(row),flush=True)
def require_allocation():
 if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Compute stages require a Slurm allocation')
def load_module(name,path):
 spec=importlib.util.spec_from_file_location(name,str(track(path)));mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
KERNEL=load_module('supp_original_metrics',G/'code/metrics_kernel.py')
def manifest():
 rows=[r for t in TASKS for r in lines(B/'manifests'/(t+'.test.seen.jsonl'))]
 assert len(rows)==3905 and len({(r['task_id'],r['input_uid']) for r in rows})==3905
 assert len({r['source_smiles'] for r in rows})==2860
 for t in TASKS:assert sum(r['task_id']==t for r in rows)==COUNTS[t]
 for r in rows:assert r['source_molecule_uid']=='raw-smiles-sha256-'+hashlib.sha256(r['source_smiles'].encode()).hexdigest()
 return rows
def subset(rows=None,profile='subset_100'):
 rows=rows or manifest();answer=[]
 for t in TASKS:
  group=sorted([r for r in rows if r['task_id']==t],key=lambda r:(hashlib.sha256(('recomb-supp-v1|'+t+'|'+r['input_uid']).encode()).hexdigest(),r['input_uid']))
  for rank,r in enumerate(group[:len(group) if profile=='full' else 2 if profile=='pilot' else 100]):
   answer.append(dict(r,subset_rank=rank,selection_hash=hashlib.sha256(('recomb-supp-v1|'+t+'|'+r['input_uid']).encode()).hexdigest()))
 assert len(answer)==(3905 if profile=='full' else 40//2 if profile=='pilot' else 996)
 return answer
def response_seed(base,backbone,task,uid,arm,node,trial,role):
 fields=dict(base_run_seed=base,backbone=backbone,task_id=task,instance_id=uid,trajectory_or_node=node,trial=trial,response_role=role)
 if arm=='common_initial':fields['shared_stream']='common_initial'
 else:fields['arm']=arm
 return int(digest(fields)[:16],16)%(2**63-1),fields

def labels(record,c):
 raw=c.get('raw_smiles',c.get('parsed_smiles'))
 if not isinstance(raw,str) or type(c.get('rdkit_valid')) is not bool:raise ValueError('Missing raw candidate or explicit validity')
 valid=c['rdkit_valid'] and bool(raw.strip());p=c.get('properties');src=record['source_properties'];names=list(src)
 KERNEL.require_source(src,names);eligible=valid and KERNEL.finite_values(p,names)
 if valid and not eligible:
  # Only explicitly recorded original chemistry/precision failure is a known failure.
  error=c.get('property_error')
  if error:
   policy=load_module('supp_known_error_check',R/'recovery_20261027_v1/property_repair.py')
   policy.known_error(error)
  flags=(bool(error) or c.get('eval_valid') is False or any(x.get('valid') is False for x in c.get('canonical_representations',[]) if x.get('environment')=='gellmo_eval_py36'))
  if not flags:raise ValueError('Unknown missing target properties on structurally valid candidate '+str(c.get('candidate_id')))
 success=bool(eligible and KERNEL.strictly_improves(src,p,names))
 delta=bool(success and KERNEL.meets_delta(src,p,names,{k:DELTAS[k] for k in names}))
 return dict(structural_valid=bool(valid),finite_complete_properties=KERNEL.finite_values(p,names),eligible=bool(eligible),success=success,threshold_success=delta)

def extracted_function(path,names,scope):
 tree=ast.parse(track(path).read_text());nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names]
 assert len(nodes)==len(names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope);return scope

def prepare():
 rows=manifest();sub=subset(rows);emit(OUT/'subset_manifest.jsonl',sub);emit(OUT/'pilot_manifest.jsonl',subset(rows,'pilot'))
 config=dict(schema='recomb-supp-v1',profile='subset_100',tasks=COUNTS,backbones=MODELS,seeds=SEEDS,arms=['GoT','Sampling with GoT scoring','Reflexion with explicit raw feedback','Reflexion without reflection'],n_inputs=996,arm_input_runs=23904,subset_sha256=sha(OUT/'subset_manifest.jsonl'),pilot=dict(inputs_per_task=2,seed=1729,use='engineering only'),K=KS,similarity_thresholds=[.2,.4,.6,.8],primary_similarity_threshold=.6,similarity_operator='>',similarity=dict(environment='gellmo_eval_py36',rdkit='2020.03.3',radius=2,bits=2048,includeChirality=False,full_precision=True),success_function=str(G/'code/metrics_kernel.py'),candidate_precision='pandas1.0.5 numpy1.18.5 round(2) CSV roundtrip in Python3.6',source_precision='original manifest',got=dict(initial=12,parents=6,refinements=6,merges=6,aggregations=4,temperature=1.,top_p=1.,top_k=0,initial_cap=200,later_cap=220),reflexion=dict(trajectories=20,max_trials=3,temperature=1.,top_p=1.,top_k=0,thought_cap=250,action_cap=250,reflection_cap=250,first_success_stops_trajectory=True,raw_feedback_explicit_in_both_arms=True),bootstrap=dict(seed=20261014,replicates=2000,unit='original/raw source SMILES',task_weights='equal',zero_denominator='redraw',pointwise=True),max_concurrent_gpus=2,original_data_read_only=True,overleaf=False)
 # JSON is a strict YAML subset and avoids a new runtime dependency.
 write(OUT/'resolved_config.yaml',config);write(OUT/'version_and_hashes.json',dict(spec=sha(CODE/'SPEC_CN.txt'),original_code={str(p):sha(p) for p in [G/'code/scorer.py',G/'code/metrics_kernel.py',G/'code/precision.py',G/'code/graph_adapter.py',G/'code/property_service.py',B/'p6_llama_main_ten_tasks_v1/src/reflexion.py',V/'voting/extract.py',B/'recomb_full_results_20261028_v1/build_statistics.py']}))
 emit(OUT/'sources_manifest.jsonl',SOURCES.values());log('prepared',inputs=len(sub),profile='subset_100')
