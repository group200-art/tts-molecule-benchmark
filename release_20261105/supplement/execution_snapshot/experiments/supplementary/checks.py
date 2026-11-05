"""Checks target rules capable of changing the scientific comparisons."""
from common import *
from stats import permutation_success,vote_uniform,constrained,Bootstrap
from itertools import permutations,product
from collections import Counter

def main(args=None):
 checks=[]
 def ok(name,condition):
  if not condition:raise AssertionError(name)
  checks.append(name)
 rows=manifest();sub=subset(rows);ok('996 fixed inputs independent of outputs',len(sub)==996 and len({r['input_uid'] for r in sub})==996)
 ok('stable subset across calls',digest(sub)==digest(subset(rows)))
 ok('twenty fixed pilot inputs',len(subset(rows,'pilot'))==20)
 ok('raw source identity is not chemical canonical identity',len({r['source_smiles'] for r in rows})==2860)
 ok('strict boundary',not constrained(True,.6,.6) and constrained(True,.604,.6))
 for sim in [None,float('nan'),float('inf')]:
  try:constrained(True,sim,.6)
  except ValueError:checks.append('missing successful similarity rejected '+str(sim))
  else:raise AssertionError('Missing similarity not rejected')
 ok('failed output need not have similarity',not constrained(False,None,.6))
 for n in range(1,6):
  for s in range(n+1):
   pool=[1]*s+[0]*(n-s)
   for k in range(1,n+1):
    exact=sum(any(p[:k]) for p in permutations(pool))/math.factorial(n)
    ok('uniform ordering exact n%d s%d k%d'%(n,s,k),abs(exact-permutation_success(n,s,k))<1e-14)
 groups=[[0,0,1],[0,1,1],[0,0,0]]
 for k in [1,2,3]:
  exact=sum(any(x) for x in product(*groups[:k]))/(3**k)
  ok('group uniform exact '+str(k),abs(vote_uniform([sum(g) for g in groups],k,3)-exact)<1e-14)
 selector=extracted_function(V/'voting/extract.py',{'count_vote'},dict(Counter=Counter))['count_vote']
 cs=[dict(class_type='VALID_CANONICAL',class_value=x,global_slot=i,rdkit_valid_ev=True,raw_response=x) for i,x in enumerate(['b','a','a','b'])]
 w,stats=selector(cs);ok('duplicate frequency counted; ties use first class appearance',w['global_slot']==0 and stats['top_votes']==2 and stats['tie_count']==2)
 src=dict(bbbp=.2);c=dict(candidate_id='x',raw_smiles='CC',rdkit_valid=True,properties=dict(bbbp=.4))
 lab=labels(dict(source_properties=src),c);ok('decimal delta boundary inclusive',lab['success'] and lab['threshold_success'])
 c['properties']={'bbbp':.2};ok('directional improvement strict',not labels(dict(source_properties=src),c)['success'])
 try:labels(dict(source_properties=src),dict(c,properties=None))
 except ValueError:checks.append('unknown missing property raises')
 else:raise AssertionError('Unknown missing property silently failed')
 from bridges import got_modules
 scorer,graph=got_modules();ok('constant feature value',scorer.minmax([1.,1.])[0]==[.5,.5])
 nodes=[dict(node_id=str(i),created_order=i,parsed_smiles=x) for i,x in enumerate(['CC','CC','','bad'])]
 props={str(i):dict(rdkit_valid=i<2,inf_canonical='CC' if i<2 else '',properties=dict(bbbp=.4,sas=2.) if i<2 else None,raw_properties=dict(sim=.7) if i<2 else None) for i in range(4)}
 score=scorer.score_pool(nodes,props,src,[dict(property='bbbp',direction=1)])
 ok('stable exact ties',score['ranked_node_ids']==['0','1','3','2'])
 ok('duplicate penalty applies to both copies',all(s['components']['duplicate_penalty']==-.25 for s in score['scores'][:2]))
 ok('empty and invalid scores',score['scores'][2]['value_score']==-200 and score['scores'][3]['value_score']==-100)
 a,_=response_seed(1729,'mistral','BDP','x','common_initial',0,0,'action');b,_=response_seed(1729,'mistral','BDP','x','other',0,0,'action');ok('independent post-branch RNG',a!=b)
 from generation import actor_prompt
 templates=read(B/'p6_llama_main_ten_tasks_v1/configs/reflexion_prompts.json');history=[dict(raw_action='<SMILES>bad</SMILES>',feedback={'observation':'Answer is INCORRECT; molecular output is invalid'})]
 for memory in [[],['Reconsider the molecular edit.']]:
  p=actor_prompt(templates,'task',history,memory,dict(raw_text='a thought'));ok('raw feedback reaches actual action prompt',all(x in p for x in ['<SMILES>bad</SMILES>','Answer is INCORRECT','Action:']))
 write(OUT/'checks.json',dict(status='PASS',checks=checks,count=len(checks),pending_runtime_checks=['actual shared initial content','all trajectory stopping and terminal ordering','property/molecular exception paths on full runs','same scorer replay on every historical graph','source bootstrap paired generation runs']))
 log('checks_passed',count=len(checks))
