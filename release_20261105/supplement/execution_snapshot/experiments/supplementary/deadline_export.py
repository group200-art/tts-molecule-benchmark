"""Export only fully verified analyses; no partial-scope numerical tables."""
from deadline_scope import *
import subprocess

NAMES=dict(beam='Beam',sampling='Sampling',self_refine='Self-Refine',tot='ToT',got='GoT',s1='s1',voting='Vote',icl='ICL',reflexion='Reflexion')

def main():
 assert read(RELEASE/'offline_audit.json')['status']=='PASS'
 rows=list(csv.DictReader((RELEASE/'offline_summary.csv').open()));ds=list(csv.DictReader((RELEASE/'offline_paired_differences.csv').open()))
 lut={(r['analysis'],r['backbone'],r['method'],r['selector'],r['metric']):r for r in rows}
 def val(a,b,m,s,k):return float(lut[a,b,m,s,k]['estimate'])
 def contrast(a,b,m,s,k):return next(r for r in ds if (r['analysis'],r['backbone'],r['method'],r['contrast'],r['metric'])==(a,b,m,s,k))
 def ci(r):return ('%+.2f [%+.2f, %+.2f]'%(float(r['estimate']),float(r['ci_low']),float(r['ci_high']))).replace('-', '$-$')
 def table(name,caption,columns,header,body,label):
  text='\n'.join([r'\begin{table}[!htbp]',r'\centering',r'\setlength{\tabcolsep}{4pt}',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\begin{tabular}{'+columns+'}',r'\toprule',header+r' \\',r'\midrule']+body+[r'\bottomrule',r'\end{tabular}',r'\end{table}'])+'\n'
  write_text(RELEASE/'tables'/(name+'.tex'),text)
 body=[]
 for b in MODELS:
  body.append(r'\multicolumn{6}{l}{\textbf{'+b.capitalize()+r'}} \\')
  metrics=['SR@1','SR_sim>0.6@1','SR@20','SR_sim>0.6@20']
  maxima={k:max(val('similarity',b,m,'',k) for m in METHODS) for k in metrics}
  for m in METHODS:
   values=[val('similarity',b,m,'',k) for k in metrics]
   d=contrast('similarity',b,m,'Sim > 0.6 minus unconstrained','SR@20')
   formatted=[r'\textbf{%.2f}'%v if v==maxima[k] else '%.2f'%v for k,v in zip(metrics,values)]
   body.append(NAMES[m]+' & '+' & '.join(formatted)+' & '+ci(d)+r' \\')
  if b==MODELS[0]:body.append(r'\midrule')
 table('table_S1_similarity',r'Success rates on the original returned sequences, with and without the strict Tanimoto $T>0.6$ requirement between a candidate and its source molecule. All 3,905 inputs and twenty original positions are retained for each configuration. Rates are percentages averaged equally over ten tasks. The difference subtracts original SR@20 from constrained SR@20, in percentage points. Brackets show paired source bootstrap pointwise 95\% intervals conditional on saved outputs. Bold marks the highest observed rate in each column and backbone, without a significance claim.','lrrrrl',r'Method & SR@1 & $T>0.6$ @1 & SR@20 & $T>0.6$ @20 & Difference [95\% CI]',body,'tab:supp-similarity')
 body=[];orders=[('same20_score_order','Score order'),('same20_creation_order','Creation order'),('same20_uniform_permutation','Uniform permutation')]
 for b in MODELS:
  body.append(r'\multicolumn{4}{l}{\textbf{'+b.capitalize()+r'}} \\')
  for s,label in orders:
   d='---' if s==orders[0][0] else ci(contrast('got_ordering',b,'got',s+' minus same20_score_order','SR@1'))
   body.append(label+' & %.2f & %.2f & '%(val('got_ordering',b,'got',s,'SR@1'),val('got_ordering',b,'got',s,'SR@20'))+d+r' \\')
  if b==MODELS[0]:body.append(r'\midrule')
 table('table_S2_got_ordering',r'GoT with the same twenty returned candidates in different orders, using all 3,905 inputs per backbone. Rates are percentages averaged equally over tasks. Uniform permutation reports its analytic expectation. The candidate multiset, SR@20 and validity are unchanged for every input. Differences are in percentage points relative to score order at $K=1$, with paired source bootstrap pointwise 95\% intervals. This diagnoses ordering within the fixed set; it does not test the contribution of generation.','lrrl',r'Order & SR@1 & SR@20 & SR@1 difference [95\% CI]',body,'tab:supp-got-order')
 body=[];vs=[('Vote','Vote'),('First proposal per group','First proposal'),('Uniform position per group','Uniform position'),('Diagnostic group success upper bound','Success upper bound')]
 for b in MODELS:
  body.append(r'\multicolumn{5}{l}{\textbf{'+b.capitalize()+r'}} \\')
  for s,label in vs:
   d='---' if s=='Vote' else ci(contrast('vote_fixed_pool',b,'voting',s+' minus Vote','SR@20'))
   v='NA' if s==vs[-1][0] else '%.2f'%val('vote_fixed_pool',b,'voting',s,'validity')
   body.append(label+' & %.2f & %.2f & '%(val('vote_fixed_pool',b,'voting',s,'SR@1'),val('vote_fixed_pool',b,'voting',s,'SR@20'))+v+' & '+d+r' \\')
  if b==MODELS[0]:body.append(r'\midrule')
 table('table_S3_vote_pool',r'Alternative selectors on the saved Vote proposal groups, retaining all 400 proposals in twenty groups per input. All 3,905 inputs are included per backbone. Rates are percentages averaged equally over tasks. First proposal retains group position zero even if empty or invalid. Uniform position is the analytic expectation for independent uniform choices of positions. The success upper bound uses outcome labels and is diagnostic, not a selector with matched information; its validity is unspecified. Differences subtract Vote SR@20 in percentage points; intervals are paired source bootstrap pointwise 95\% intervals.','lrrrl',r'Selector & SR@1 & SR@20 & Validity & SR@20 difference [95\% CI]',body,'tab:supp-vote-pool')
 generation=False
 state=RELEASE/'generation/status.json'
 if state.exists() and read(state)['status']=='PASS_COMPLETE_FIXED_SCOPE':
  assert read(state)['inputs_per_backbone']==500 and read(state)['arm_inputs']==2000
  gs=list(csv.DictReader((RELEASE/'generation/per_seed_summary.csv').open()));gp=list(csv.DictReader((RELEASE/'generation/paired_differences.csv').open()));work=list(csv.DictReader((RELEASE/'generation/workload.csv').open()))
  gl={(r['backbone'],r['arm'],r['metric']):float(r['estimate']) for r in gs};body=[]
  for b in MODELS:
   body.append(r'\multicolumn{7}{l}{\textbf{'+b.capitalize()+r'}} \\')
   for arm,label in [('GoT','GoT'),('Sampling with GoT scoring','Sampling, same scorer')]:
    ww=[r for r in work if r['backbone']==b and r['arm']==arm];assert len(ww)==500
    vals=[gl[b,arm,k] for k in ['SR@1','SR@20','SR_sim>0.6@20','validity']]
    body.append(label+' & '+' & '.join('%.2f'%v for v in vals)+' & %.0f & %.0f'%(sum(float(r['input_tokens']) for r in ww)/500,sum(float(r['output_tokens']) for r in ww)/500)+r' \\')
   if b==MODELS[0]:body.append(r'\midrule')
  table('table_S4_got_generation',r'GoT and independent Sampling with the same molecular scorer on the fixed 500-input subset (50 per task), both backbones, and generation seed 1729. Each pair actually shares twelve initial candidates, then generates sixteen additional candidates per arm; each complete pool is scored separately and its top twenty returned. The arms use the same scoring code, temperature, candidate caps and evaluation precision. Rates are percentages with equal task weights. Mean input and output tokens per input count the shared initial generation in each arm. Intervals are conditional on this single generation run; no three-seed stability claim is made.','lrrrrrr',r'Configuration & SR@1 & SR@20 & $>0.6$ @20 & Validity & \shortstack{Input\\tokens} & \shortstack{Output\\tokens}',body,'tab:supp-got-generation')
  body=[]
  for b in MODELS:
   for k in ['SR@1','SR@20','SR_sim>0.6@20']:
    d=next(r for r in gp if (r['backbone'],r['metric'])==(b,k));label=k.replace('SR_sim>0.6@20',r'SR@20, Sim $>0.6$')
    body.append(b.capitalize()+' & '+label+' & '+ci(d)+r' \\')
  table('table_S5_got_differences',r'Paired GoT minus Sampling differences in percentage points for the complete 500-input, one-seed comparison. SR@1 is the prespecified primary endpoint. Pointwise 95\% intervals resample original source identifiers jointly across tasks and arms; they do not quantify variability across generation seeds.','lll',r'Backbone & Endpoint & Difference [95\% CI]',body,'tab:supp-got-difference');generation=True
 cache=RELEASE/'plot_cache';cache.mkdir(exist_ok=True);os.environ['MPLCONFIGDIR']=str(cache)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
 figs=[]
 for name,analysis,method,selectors in [('got_same_twenty','got_ordering','got',[(s,l) for s,l in orders]),('vote_fixed_pool','vote_fixed_pool','voting',[(s,('Uniform position (analytic)' if s==vs[2][0] else 'Success upper bound (diagnostic)' if s==vs[3][0] else l)) for s,l in vs])]:
  fig,axes=plt.subplots(1,2,figsize=(6.5,3.7),sharey=True)
  for b,ax in zip(MODELS,axes):
   for s,label in selectors:ax.plot(KS,[val(analysis,b,method,s,'SR@'+str(k)) for k in KS],marker='o',label=label)
   ax.set_title(b.capitalize());ax.set_xticks(KS);ax.set_xlabel('Outputs evaluated');ax.set_ylim(0,103)
  axes[0].set_ylabel('Mean success rate (%)')
  fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,1.02),ncol=2,frameon=False,fontsize=10)
  fig.tight_layout(rect=(0,0,1,.87))
  for ext in ['pdf','png']:
   p=RELEASE/'figures'/(name+'.'+ext);p.parent.mkdir(exist_ok=True);fig.savefig(p,dpi=180,bbox_inches='tight');figs.append(str(p))
  plt.close(fig)
 names=['table_S1_similarity','table_S2_got_ordering','table_S3_vote_pool']+(['table_S4_got_generation','table_S5_got_differences'] if generation else [])
 tex=[r'\documentclass[10pt,letterpaper]{article}',r'\usepackage[margin=1in]{geometry}',r'\usepackage[T1]{fontenc}',r'\usepackage{lmodern,booktabs,graphicx,amsmath}',r'\usepackage[hidelinks]{hyperref}',r'\usepackage{float}',r'\begin{document}',r'\renewcommand{\thetable}{S\arabic{table}}',r'\renewcommand{\thefigure}{S\arabic{figure}}',r'\section*{Verified supplementary results}',r'All analyses retain the fixed input denominators. The original manuscript and primary results are unchanged. These tables are prepared separately for author review.']
 for name in names:
  tex.append(r'\input{tables/'+name+'}')
  if name in ['table_S1_similarity','table_S3_vote_pool','table_S5_got_differences']:tex.append(r'\clearpage')
 for name in ['got_same_twenty','vote_fixed_pool']:tex.extend([r'\begin{figure}[!htbp]\centering',r'\includegraphics[width=\linewidth]{figures/'+name+'}',r'\caption{'+('Ordering of the same twenty GoT candidates.' if name=='got_same_twenty' else 'Selectors on the saved Vote groups. Random selection is an analytic expectation; the upper bound uses outcome labels.')+'}',r'\end{figure}'])
 tex.append(r'\end{document}');write_text(RELEASE/'verified_tables.tex','\n'.join(tex)+'\n')
 env=dict(os.environ)
 for k,d in [('TEXMFVAR','var'),('TEXMFCONFIG','config')]:
  p=RELEASE/'tex_cache'/d;p.mkdir(parents=True,exist_ok=True);env[k]=str(p)
 for run in range(2):
  r=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','-no-shell-escape','verified_tables.tex'],cwd=RELEASE,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  write_text(RELEASE/('tables_compile_%d.log'%run),r.stdout)
  assert r.returncode==0,r.stdout[-2000:]
 save(RELEASE/'export_status.json',dict(status='EXPORTED_VERIFIED_ANALYSES',offline=True,generation=generation,generation_scope='500 inputs per backbone, one seed' if generation else 'not included until complete',tables=names,figures=figs,pdf=str(RELEASE/'verified_tables.pdf'),created_utc=now().isoformat(),original_paper_modified=False,overleaf_pushed=False))
 print('Exported tables; generation included:',generation,flush=True)

if __name__=='__main__':main()
