"""Export measured supplementary tables/figures; never touch the manuscript."""
from common import *

def main(args=None):
 cache=OUT/'plot_cache';cache.mkdir(parents=True,exist_ok=True);os.environ['MPLCONFIGDIR']=str(cache)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
 source=OUT/'offline_summary.csv'
 if not source.exists():raise RuntimeError('Run aggregate first')
 rows=list(csv.DictReader(source.open()));lookup={(r['analysis'],r['backbone'],r['method'],('' if r['selector']=='NA' else r['selector']),r['metric']):r for r in rows}
 def get(analysis,b,m,selector,metric):return lookup[analysis,b,m,selector,metric]
 def number(r,key='estimate'):return float(r[key]) if r[key] not in ('NA','') else None
 def finish(fig,name):
  fig.tight_layout();safe(OUT/'figures'/(name+'.pdf'));fig.savefig(OUT/'figures'/(name+'.pdf'),bbox_inches='tight');fig.savefig(OUT/'figures'/(name+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)
 diffs=list(csv.DictReader((OUT/'offline_paired_differences.csv').open()))
 names=dict(beam='Beam',sampling='Sampling',self_refine='Self-Refine',tot='ToT',got='GoT',s1='s1',voting='Vote',icl='ICL',reflexion='Reflexion')
 if all(('similarity',b,m,'','SR@20') in lookup for b in MODELS for m in METHODS):
  fig,axes=plt.subplots(1,2,figsize=(10.5,4.1),sharey=True)
  for b,ax in zip(MODELS,axes):
   x=list(range(9));base=[number(get('similarity',b,m,'','SR@20')) for m in METHODS];con=[number(get('similarity',b,m,'','SR_sim>0.6@20')) for m in METHODS]
   ax.bar([z-.18 for z in x],base,width=.36,yerr=[[number(get('similarity',b,m,'','SR@20'))-number(get('similarity',b,m,'','SR@20'),'ci_low') for m in METHODS],[number(get('similarity',b,m,'','SR@20'),'ci_high')-number(get('similarity',b,m,'','SR@20')) for m in METHODS]],error_kw={'elinewidth':.6,'capsize':1.5},label='Original SR@20',color='#6888A3');ax.bar([z+.18 for z in x],con,width=.36,yerr=[[number(get('similarity',b,m,'','SR_sim>0.6@20'))-number(get('similarity',b,m,'','SR_sim>0.6@20'),'ci_low') for m in METHODS],[number(get('similarity',b,m,'','SR_sim>0.6@20'),'ci_high')-number(get('similarity',b,m,'','SR_sim>0.6@20')) for m in METHODS]],error_kw={'elinewidth':.6,'capsize':1.5},label='SR@20 with similarity > 0.6',color='#CC8053');ax.set_xticks(x, [names[m] for m in METHODS],rotation=55,ha='right');ax.set_title(b.capitalize());ax.set_ylim(0,103);ax.set_ylabel('Mean success rate across tasks (%)')
  fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=2,bbox_to_anchor=(.5,1.08),frameon=False,fontsize=9);finish(fig,'similarity_constraint')
  tex=['% Generated from similarity_constraints.csv; separate supplementary artifact.',r'\begin{table}[t]',r'\centering',r'\caption{Success rates with and without a strict Tanimoto similarity requirement. Entries are means over ten tasks in percent, using all 3,905 inputs per configuration. Differences subtract unconstrained SR@20 and use paired source bootstrap pointwise 95\% intervals. Outputs and order remain fixed.}',r'\begin{tabular}{llrrrrl}',r'\toprule',r'Backbone & Method & SR@1 & Sim $>0.6$ @1 & SR@20 & Sim $>0.6$ @20 & Difference [95\% CI] \\',r'\midrule']
  for b in MODELS:
   for m in METHODS:
    vals=[number(get('similarity',b,m,'',metric)) for metric in ['SR@1','SR_sim>0.6@1','SR@20','SR_sim>0.6@20']]
    d=next(r for r in diffs if r['analysis']=='similarity' and r['backbone']==b and r['method']==m)
    tex.append(b.capitalize()+' & '+names[m]+' & '+' & '.join('%.2f'%v for v in vals)+' & %.2f [%.2f, %.2f]'%(float(d['estimate']),float(d['ci_low']),float(d['ci_high']))+r' \\')
   tex.append(r'\midrule')
  tex+= [r'\bottomrule',r'\end{tabular}',r'\end{table}'];safe(OUT/'tables/table_A_similarity.tex').write_text('\n'.join(tex)+'\n')
  fig,axes=plt.subplots(1,2,figsize=(9.5,3.5),sharey=True)
  for b,ax in zip(MODELS,axes):
   for m in METHODS:ax.plot([0,.2,.4,.6,.8],[number(get('similarity',b,m,'','SR@20'))]+[number(get('similarity',b,m,'','SR_sim>'+str(t)+'@20')) for t in [.2,.4,.6,.8]],marker='o',markersize=3,label=names[m])
   ax.set_xticks([0,.2,.4,.6,.8],['None','>0.2','>0.4','>0.6','>0.8']);ax.set_title(b.capitalize());ax.set_xlabel('Similarity constraint');ax.set_ylabel('Mean SR@20 across tasks (%)')
  axes[-1].legend(fontsize=7,ncol=2);finish(fig,'similarity_thresholds')
 selectors=['Vote','First proposal per group','Uniform position per group','Diagnostic group success upper bound']
 if all(('vote_fixed_pool',b,'voting',s,'SR@20') in lookup for b in MODELS for s in selectors):
  fig,axes=plt.subplots(1,2,figsize=(9,3.5),sharey=True)
  labels=['Vote','First proposal in each group','Uniform position (analytic expectation)','Group success upper bound (diagnostic)']
  for b,ax in zip(MODELS,axes):
   for s,label,style in zip(selectors,labels,['-','--','-.',':']):ax.plot(KS,[number(get('vote_fixed_pool',b,'voting',s,'SR@'+str(k))) for k in KS],marker='o',ls=style,label=label)
   ax.set_xticks(KS);ax.set_ylim(0,103);ax.set_title(b.capitalize());ax.set_xlabel('Number of returned outputs evaluated');ax.set_ylabel('Mean success rate across tasks (%)')
  axes[-1].legend(fontsize=7);finish(fig,'vote_fixed_pool')
 selectors=['same20_score_order','same20_creation_order','same20_uniform_permutation']
 if all(('got_ordering',b,'got',s,'SR@20') in lookup for b in MODELS for s in selectors):
  fig,axes=plt.subplots(1,2,figsize=(9,3.3),sharey=True)
  for b,ax in zip(MODELS,axes):
   for s,label in zip(selectors,['Original score order','Creation order','Uniform permutation (analytic expectation)']):ax.plot(KS,[number(get('got_ordering',b,'got',s,'SR@'+str(k))) for k in KS],marker='o',label=label)
   ax.set_xticks(KS);ax.set_title(b.capitalize());ax.set_xlabel('Number of returned outputs evaluated');ax.set_ylabel('Mean success rate across tasks (%)')
  axes[-1].legend(fontsize=7);finish(fig,'got_same_twenty_ordering')
  fig,ax=plt.subplots(figsize=(6.5,3));left=[0,0]
  for metric,label in [('pool_has_no_success','No successful candidate in all 28'),('selection_omits_success','Success omitted from returned 20'),('success_returned_but_not_first','Success returned, but not first')]:
   vals=[number(get('got_ordering',b,'got','first_output_failure_decomposition',metric)) for b in MODELS];ax.barh([0,1],vals,left=left,label=label);left=[l+v for l,v in zip(left,vals)]
  ax.set_yticks([0,1],[b.capitalize() for b in MODELS]);ax.set_xlabel('Mean fraction of inputs across tasks (%)');ax.legend(fontsize=8);finish(fig,'got_first_output_failure_decomposition')
 p=OUT/'generation_summary.csv';gen=list(csv.DictReader(p.open())) if p.exists() else []
 if gen:
  paired=list(csv.DictReader((OUT/'paired_differences.csv').open()));work=list(csv.DictReader((OUT/'workload.csv').open()));seeds=list(csv.DictReader((OUT/'per_seed_summary.csv').open()))
  gl={(r['backbone'],r['arm'],r['metric']):r for r in gen}
  primary=[r for r in paired if r['seed']=='mean_of_three' and r['primary']=='True']
  tex=['% Three-seed results, separate supplementary output.',r'\begin{table}[t]',r'\centering',r'\caption{Matched generation controls on the fixed 996-input subset. Success and validity are averaged over ten tasks within each seed, then over three seeds. Output tokens use equal input weights and include shared initial generation in each arm. Paired differences use the primary endpoint specified for each comparison; pointwise intervals are conditional on the three completed generation runs.}',r'\begin{tabular}{llrrrr}',r'\toprule',r'Backbone & Configuration & SR@1 & SR@20 & Validity & Output tokens \\',r'\midrule']
  arms=['GoT','Sampling with GoT scoring','Reflexion with explicit raw feedback','Reflexion without reflection']
  for b in MODELS:
   for arm in arms:
    values=[float(gl[b,arm,k]['estimate']) for k in ['SR@1','SR@20','validity']];cost=[float(r['output_tokens']) for r in work if (r['backbone'],r['arm'])==(b,arm)]
    tex.append(b.capitalize()+' & '+arm+' & '+' & '.join('%.2f'%v for v in values)+' & %.0f'%(sum(cost)/len(cost))+r' \\')
  tex += [r'\bottomrule',r'\end{tabular}',r'\par\medskip',r'\begin{tabular}{llrrl}',r'\toprule',r'Backbone & Comparison & Endpoint & Difference & 95\% CI \\',r'\midrule']
  for r in primary:
   label='GoT minus Sampling' if r['arm']=='GoT' else 'Reflection minus no reflection'
   tex.append(r['backbone'].capitalize()+' & '+label+' & '+r['metric']+' & %.2f & [%.2f, %.2f]'%(float(r['estimate']),float(r['ci_low']),float(r['ci_high']))+r' \\')
  tex += [r'\bottomrule',r'\end{tabular}',r'\end{table}'];safe(OUT/'tables/table_B_generation.tex').write_text('\n'.join(tex)+'\n')
  fig,axes=plt.subplots(1,2,figsize=(9,3.5))
  for ax,(arm,metric,title) in zip(axes,[('GoT','SR@1','GoT minus Sampling with GoT scoring'),('Reflexion with explicit raw feedback','SR@20','Reflection minus no reflection')]):
   for y,b in enumerate(MODELS):
    r=next(r for r in primary if r['backbone']==b and r['arm']==arm);mid=float(r['estimate']);lo=float(r['ci_low']);hi=float(r['ci_high'])
    ax.hlines(y,lo,hi,color='#335F7D');ax.plot([lo,hi],[y,y],marker='|',linestyle='None',color='#335F7D');ax.plot(mid,y,marker='o',color='#335F7D')
   ax.axvline(0,color='gray',lw=.8,ls='--');ax.set_yticks([0,1],[b.capitalize() for b in MODELS]);ax.set_ylim(-.5,1.5);ax.set_title(title,fontsize=9);ax.set_xlabel('Difference in mean '+metric+' (percentage points)')
  finish(fig,'matched_generation_primary_contrasts')
  tex=['% Per-seed primary contrasts; no seed/input flattening.',r'\begin{tabular}{llrrrrr}',r'\toprule',r'Backbone & Comparison & Seed 1729 & Seed 2718 & Seed 3141 & Range & SD \\',r'\midrule']
  for r in primary:
   specific=[next(float(s['estimate']) for s in paired if s['backbone']==r['backbone'] and s['arm']==r['arm'] and s['metric']==r['metric'] and s['seed']==str(seed)) for seed in SEEDS]
   label='GoT (SR@1)' if r['arm']=='GoT' else 'Reflection (SR@20)'
   tex.append(r['backbone'].capitalize()+' & '+label+' & '+' & '.join('%.2f'%v for v in specific)+' & [%.2f, %.2f] & %.2f'%(float(r['seed_difference_min']),float(r['seed_difference_max']),float(r['seed_difference_sample_sd']))+r' \\')
  tex += [r'\bottomrule',r'\end{tabular}'];safe(OUT/'tables/generation_seed_contrasts.tex').write_text('\n'.join(tex)+'\n')
 write(OUT/'figure_manifest.json',dict(status='EXPORTED_AVAILABLE_COMPLETED_ANALYSES',files=[dict(path=str(p),sha256=sha(p)) for p in sorted((OUT/'figures').glob('*'))],generation_table='complete' if gen else 'pending generation; no numerical placeholders exported'))
 log('figures_exported',generation_available=bool(gen))
if __name__=='__main__':main()
