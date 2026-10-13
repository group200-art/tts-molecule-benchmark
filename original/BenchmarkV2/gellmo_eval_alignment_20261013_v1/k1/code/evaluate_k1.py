"""Supplement original-GeLLMO metrics for frozen K=1 endpoints; CPU only.

Python 3.6 compatible. Never selects or generates a different molecular output.
Run in gellmo_eval_py36 to reproduce the retained legacy chemistry environment.
"""
import csv
import gzip
import hashlib
import io
import json
import math
import os
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[2]
SRC = OUT / 'sources'
sys.path.insert(0, str(ROOT / 'data/props'))
import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
import sascorer
RDLogger.DisableLog('rdApp.*')

TASKS = ['BDP','BDQ','BPQ','DPQ','BDPQ','MPQ','BDMQ','BHMQ','BMPQ','HMPQ']
MODELS = ['gellmo_p6_mistral','gellmo_p6_llama']
METHODS = ['beam20','sampling_verifier20','self_refine_r1','tot_lm_vote','got','s1','ev','fewshot','reflexion']
COUNTS = dict(zip(TASKS, [500]*7+[118,191,96]))
REFERENCE = ROOT / 'data/unique_mols_in_one_ds_pairs.txt'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def read(path):
    with Path(path).open() as f: return json.load(f)

def save(name, obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')

def csvwrite(name, rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def avg(values):
    values=list(values)
    return math.fsum(values)/len(values) if values else None

def close(a,b):
    return a is not None and b is not None and math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-11)

def fp(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=2048,useChirality=False)

def official_ri(source, values):
    # Exact public evaluate.ipynb cell 11 branch, NOT stabilized RI_0.1.
    return avg(abs(values[p]-x)/abs(x) if x != 0 else (values[p]-.1)/.1 for p,x in source.items())

def main():
    assert rdkit.__version__=='2020.03.3'
    assert np.__version__=='1.18.5' and pd.__version__=='1.0.5'
    assert not os.environ.get('CUDA_VISIBLE_DEVICES','')
    snapshots=read(SRC/'snapshot_manifest.json')
    for name,e in snapshots.items(): assert sha(SRC/name)==e['sha256']==sha(e['source_path'])
    audit=read(SRC/'endpoint_audit.json');manifests=read(SRC/'manifests.json')
    assert audit['status']=='COMPLETE_FULL3905_TWO_MODELS_NINE_METHODS_K1'
    original=read(SRC/'evaluate.ipynb')
    selector_source=''.join(original['cells'][6]['source'])
    namespace={};exec(selector_source,namespace)
    original_selector=namespace['find_best_optimized']
    reference_literal=set(np.loadtxt(str(REFERENCE),dtype=str,comments=None))
    reference_path=OUT/'sources/novelty_reference_canonical_rdkit2020.txt'
    reference_canonical=set(); reference_invalid=0; reference_changed=0
    if reference_path.exists() and (OUT/'metric_contract.json').exists():
        previous=read(OUT/'metric_contract.json')
        assert previous['environment']['rdkit']==rdkit.__version__
        assert previous['reference']['sha256']==sha(REFERENCE)
        assert previous['reference']['canonical_snapshot_sha256']==sha(reference_path)
        reference_canonical=set(reference_path.read_text().splitlines())
        reference_invalid=previous['reference']['invalid_strings']
        reference_changed=previous['reference']['canonical_changed']
        assert len(reference_canonical)==previous['reference']['canonical_count']
        print('reused verified reference canonicalization',len(reference_canonical),flush=True)
    else:
        for i,smi in enumerate(sorted(reference_literal)):
            m=Chem.MolFromSmiles(smi)
            if m is None: reference_invalid+=1;continue
            can=Chem.MolToSmiles(m)
            reference_canonical.add(can);reference_changed+=int(can!=smi)
            if (i+1)%50000==0: print('reference',i+1,flush=True)
        reference_path.write_text('\n'.join(sorted(reference_canonical))+'\n')
    environment=dict(python=sys.version,platform=platform.platform(),rdkit=rdkit.__version__,numpy=np.__version__,pandas=pd.__version__,slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    contract=dict(version='GELLMO-K1-OFFICIAL-METRICS-v1',final_k=1,endpoint_changes=0,
      scope='Same 70290 fixed final outputs; not a K20 experiment and not a raw-text re-extraction.',
      SR='Strict all-target directional improvement from frozen candidate round2/CSV values against unrounded stored source values; chemical and finite-property eligibility unchanged. Independently recomputed and compared with current SR.',
      validity='Report RDKit 2020.03.3 parse of the unchanged selected SMILES; also retain current stored validity and legacy property-row eligibility separately.',
      quality_subset='Only directionally successful fixed terminal outputs; one candidate per input, no post-hoc replacement. Per-task success-conditional means, followed by equal-task macro.',
      Sim='Unrounded Tanimoto of raw source and raw selected SMILES: binary Morgan radius2, 2048 bits, useChirality=False, RDKit2020.03.3.',
      SAS='Unrounded retained data/props/sascorer.py calculateScore on selected molecule in RDKit2020.03.3; cached round2 values retained separately.',
      RI='Mean across target properties of abs(output-source)/abs(source) when source!=0, else (output-0.1)/0.1. Public notebook implementation; no normalization and no denominator flooring.',
      RI_paper_difference='Printed paper formula and public code differ; RI_official_code names code branch explicitly. RI0.1 remains a separately labeled benchmark extension.',
      Novelty='Primary Nov_official: Chem.MolToSmiles(selected) not in literal unique_mols_in_one_ds_pairs.txt set, reproducing original process-output + evaluate identity pipeline. Nov_raw_literal diagnostic uses raw selected string. Nov_canonical_reference sensitivity canonicalizes both sides in legacy RDKit.',
      APS='Per target-property arithmetic mean over success-conditional selected outputs at retained comparison precision. Report separately by property, never mix units.',
      aggregation='Task rows; IND equal mean of five task metrics, OOD equal mean of five, Macro10 equal mean of ten. APS aggregates only tasks targeting that same property, with task count explicit.',
      zeros='A task with no successful outputs would have undefined quality (null), not zero; all current groups have successes.',
      evaluator_repairs='Preserve endpoint identity; never emulate original filtered-array index bug. K1 avoids multi-candidate set ordering and tie ambiguity.',
      reuse='No target-property predictor runs; target values reused after input/source/hash and stored comparison-CSV checks.',environment=environment,
      reference=dict(path=str(REFERENCE),sha256=sha(REFERENCE),literal_count=len(reference_literal),canonical_count=len(reference_canonical),invalid_strings=reference_invalid,canonical_changed=reference_changed,canonical_snapshot_path=str(reference_path),canonical_snapshot_sha256=sha(reference_path)))
    save('metric_contract.json',contract)
    input_lookup={}
    for task,spec in manifests.items():
        assert sha(spec['path'])==spec['sha256']
        entries=[json.loads(l) for l in Path(spec['path']).read_text().splitlines() if l.strip()]
        assert len(entries)==COUNTS[task]
        for r in entries:input_lookup[task,r['input_uid']]=r
    expected_metrics={(r['model_id'],r['method_id'],r['task_id']):r['metrics'] for r in read(SRC/'task_metrics.json')}
    groups=defaultdict(list);seen=set();counts=Counter();examples=defaultdict(list)
    source_fp={};molecule_cache={};sim_cache={};raw_sha=hashlib.sha256()
    gzpath=OUT/'per_input_metrics.jsonl.gz'
    with gzpath.open('wb') as binary:
      with gzip.GzipFile(fileobj=binary,mode='wb',mtime=0) as gz:
       with io.TextIOWrapper(gz,encoding='utf-8') as sink:
        for path,digest in sorted(audit['source_files'].items()):
          assert sha(path)==digest
          with gzip.open(path,'rt') as f:
           for line in f:
            r=json.loads(line);c=r['selected_candidate'];e=r['evaluation']
            model,method,task,uid=(r[k] for k in ['model_id','method_id','task_id','input_uid'])
            key=(model,method,task,uid);assert key not in seen;seen.add(key)
            source=input_lookup[task,uid]
            assert r['source_properties']==source['source_properties']
            if 'source_smiles' in c:assert c['source_smiles']==source['source_smiles']
            assert e['n_slots']==1 and e['candidates']==[c] and c['original_slot_id']==0
            smi=c['raw_smiles'];src=r['source_properties'];props=c.get('properties')
            present=isinstance(props,dict)
            finite=present and all(isinstance(props.get(p),(int,float)) and math.isfinite(props[p]) for p in src)
            current_valid=bool(c['rdkit_valid'] and smi.strip())
            success=bool(current_valid and finite and all((props[p]<v) if p=='mutagenicity' else (props[p]>v) for p,v in src.items()))
            assert success==e['SR_dir_success']
            if finite:
                directional=all((props[p]<v) if p=='mutagenicity' else (props[p]>v) for p,v in src.items())
                assert original_selector(src,{p:[props[p]] for p in src},list(src))==(0 if directional else None)
            if c.get('rounding_csv') and present:
                compared=pd.read_csv(io.StringIO(c['rounding_csv'])).iloc[0].to_dict()
                assert all(close(compared[p],props[p]) for p in src if p in compared)
                counts['rounding_csv_checked']+=1
            if smi not in molecule_cache:
                mol=Chem.MolFromSmiles(smi) if smi.strip() else None
                molecule_cache[smi]=(mol,Chem.MolToSmiles(mol) if mol is not None else None,None,None)
            mol,canonical,molfp,sas=molecule_cache[smi]
            legacy_valid=mol is not None
            counts['current_vs_legacy_valid_mismatch']+=int(current_valid!=legacy_valid)
            if current_valid!=legacy_valid and len(examples['validity_discrepancies'])<30:examples['validity_discrepancies'].append(dict(key=key,smiles=smi,current_valid=current_valid,legacy_valid=legacy_valid))
            row=dict(model_id=model,method_id=method,task_id=task,input_uid=uid,input_idx=r['input_idx'],candidate_id=c['candidate_id'],raw_smiles=smi,canonical_smiles_legacy=canonical,source_smiles=source['source_smiles'],source_properties=src,candidate_comparison_properties={p:props[p] for p in src} if finite else None,success=success,validity_legacy_rdkit=legacy_valid,validity_current=current_valid,property_row_exists=bool(c.get('property_row_exists')),finite_target_properties=bool(finite),RI_official_code=None,RI_legacy_cached=e.get('RI_legacy'),RI_eps01_current=e.get('RI_abs_eps01'),Sim_raw=None,SAS_raw=None,Sim_current_cached=props.get('sim') if present else None,SAS_current_cached=props.get('sas') if present else None,Nov_official=None,Nov_raw_literal=None,Nov_canonical_reference=None)
            if success:
                assert legacy_valid,'Successful cached output is not valid in legacy evaluation environment'
                if molfp is None:molfp=fp(mol)
                if sas is None:sas=float(sascorer.calculateScore(mol))
                molecule_cache[smi]=(mol,canonical,molfp,sas)
                insmi=source['source_smiles']
                if insmi not in source_fp:
                    source_mol=Chem.MolFromSmiles(insmi);assert source_mol is not None;source_fp[insmi]=fp(source_mol)
                pair=(insmi,smi)
                if pair not in sim_cache:sim_cache[pair]=float(DataStructs.TanimotoSimilarity(source_fp[insmi],molfp))
                row.update(RI_official_code=official_ri(src,props),Sim_raw=sim_cache[pair],SAS_raw=sas,Nov_official=int(canonical not in reference_literal),Nov_raw_literal=int(smi not in reference_literal),Nov_canonical_reference=int(canonical not in reference_canonical))
                counts['success']+=1
                counts['source_zero_target_values']+=sum(v==0 for v in src.values())
                counts['negative_source_target_values']+=sum(v<0 for v in src.values())
                counts['RI_legacy_cached_mismatches']+=int(not close(row['RI_official_code'],e.get('RI_legacy')))
                counts['novelty_official_vs_canonical_mismatches']+=int(row['Nov_official']!=row['Nov_canonical_reference'])
                counts['novelty_official_vs_raw_literal_mismatches']+=int(row['Nov_official']!=row['Nov_raw_literal'])
                for metric,rawkey in [('Sim_raw','sim'),('SAS_raw','sas')]:
                    cachedraw=c.get('raw_properties',{}).get(rawkey) if isinstance(c.get('raw_properties'),dict) else None
                    if isinstance(cachedraw,(int,float)) and math.isfinite(cachedraw):
                        counts[metric+'_raw_cache_available']+=1
                        counts[metric+'_raw_cache_mismatches']+=int(not close(row[metric],cachedraw))
                        if not close(row[metric],cachedraw) and len(examples[metric+'_raw_cache_mismatch_examples'])<10:examples[metric+'_raw_cache_mismatch_examples'].append(dict(key=key,recomputed=row[metric],raw_cache=cachedraw))
            groups[model,method,task].append(row)
            payload=json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n';sink.write(payload);raw_sha.update(payload.encode())
          print('processed',Path(path).name,len(seen),flush=True)
    assert len(seen)==70290 and len(groups)==180
    taskrows=[];apsrows=[];quality=['RI_official_code','RI_legacy_cached','RI_eps01_current','Sim_raw','SAS_raw','Sim_current_cached','SAS_current_cached']
    novelty=['Nov_official','Nov_raw_literal','Nov_canonical_reference']
    sr_mismatches=[]
    for key in sorted(groups):
        rows=groups[key];good=[r for r in rows if r['success']];old=expected_metrics[key]
        assert len(rows)==COUNTS[key[2]] and len(good)>0
        result=dict(zip(['model_id','method_id','task_id'],key));result.update(n_inputs=len(rows),n_success=len(good),SR=100.*len(good)/len(rows),n_valid_legacy_rdkit=sum(r['validity_legacy_rdkit'] for r in rows),n_valid_current=sum(r['validity_current'] for r in rows),n_property_rows=sum(r['property_row_exists'] for r in rows),n_finite_target_rows=sum(r['finite_target_properties'] for r in rows))
        result.update(Val_legacy_rdkit=100.*result['n_valid_legacy_rdkit']/len(rows),Val_current=100.*result['n_valid_current']/len(rows),Val_property_row_diagnostic=100.*result['n_property_rows']/len(rows))
        result.update({name:avg(r[name] for r in good) for name in quality})
        result.update({name:100.*avg(r[name] for r in good) for name in novelty})
        if len(good)!=old['n_success_dir'] or not close(result['SR'],old['SR_dir']):sr_mismatches.append(key)
        for metric,currentkey in [('RI_eps01_current','RI_abs_eps01'),('Sim_current_cached','Sim_cached'),('SAS_current_cached','SAS_cached')]:assert close(result[metric],old[currentkey]),(key,metric)
        for p in good[0]['source_properties']:
            av=avg(r['candidate_comparison_properties'][p] for r in good)
            assert close(av,old['APS'][p])
            apsrows.append(dict(model_id=key[0],method_id=key[1],task_id=key[2],property=p,direction='minimize' if p=='mutagenicity' else 'maximize',n_inputs=len(rows),n_success=len(good),APS=av))
        taskrows.append(result)
    assert not sr_mismatches
    assert counts['RI_legacy_cached_mismatches']==0
    macrorows=[];apsmacros=[]
    for model in MODELS:
      for method in METHODS:
       for scope,tasks in [('IND',TASKS[:5]),('OOD',TASKS[5:]),('Macro10',TASKS)]:
        relevant=[r for r in taskrows if r['model_id']==model and r['method_id']==method and r['task_id'] in tasks]
        assert len(relevant)==len(tasks)
        macro=dict(model_id=model,method_id=method,scope=scope,n_tasks=len(tasks),n_inputs=sum(r['n_inputs'] for r in relevant),n_success=sum(r['n_success'] for r in relevant))
        macro.update({k:avg(r[k] for r in relevant) for k in ['SR','Val_legacy_rdkit','Val_current','Val_property_row_diagnostic']+quality+novelty})
        macrorows.append(macro)
        for p in sorted(set(r['property'] for r in apsrows)):
            rr=[r for r in apsrows if r['model_id']==model and r['method_id']==method and r['task_id'] in tasks and r['property']==p]
            if rr:apsmacros.append(dict(model_id=model,method_id=method,scope=scope,property=p,direction='minimize' if p=='mutagenicity' else 'maximize',n_target_tasks=len(rr),APS_equal_target_task_mean=avg(r['APS'] for r in rr)))
    csvwrite('task_metrics.csv',taskrows);csvwrite('macro_metrics.csv',macrorows);csvwrite('aps_by_task.csv',apsrows);csvwrite('aps_by_property_macro.csv',apsmacros)
    save('metrics.json',dict(task_metrics=taskrows,macro_metrics=macrorows,APS_by_task=apsrows,APS_by_property_macro=apsmacros))
    save('compatibility_examples.json',dict(examples))
    (OUT/'README.md').write_text('''# GeLLMO evaluation alignment for unchanged K=1 outputs\n\nCompleted supplemental evaluation of both models, nine methods and all 3,905 task-input pairs per configuration. No endpoint, target-property value, or source manuscript was changed; no model generation or property-predictor inference was run. This is K=1, not a K=20 replacement.\n\n`metric_contract.json` freezes definitions, the legacy chemistry environment and novelty identity policies. `task_metrics.csv` contains all 180 task groups; `macro_metrics.csv` includes IND/OOD and equal-task Macro10. Original-code RI is distinct from current stabilized RI. Sim/SAS are recomputed without two-decimal rounding; current rounded measurements remain separate columns. Novelty's primary pipeline uses canonical generated SMILES against the original literal reference set, with raw-string and canonical-reference sensitivities explicitly separated. `aps_by_task.csv` and `aps_by_property_macro.csv` never mix different property units. All quality measures condition on successful fixed outputs within each task.\n\n`per_input_metrics.jsonl.gz` preserves identities and all derived values; `audit.json` verifies coverage, unchanged success rates, hashes, original cached metric agreement, and implementation differences. Legacy-RDKit validity can differ from current recorded validity because the chemistry versions differ; neither changes the frozen SR or endpoint. The property-row diagnostic is not a substitute for RDKit validity. The original evaluator's filtered-index mapping bug is not reproduced. Full-code literal pipeline equivalence is not claimed: current endpoint parsing is retained, and property predictors are not rerun after canonicalization.\n\nReproduce on one CPU via `code/run_cpu.sbatch`, using the fixed legacy Python environment. Source snapshots and their original hashes are recorded under `sources/`.\n''')
    outputs=['metric_contract.json','per_input_metrics.jsonl.gz','task_metrics.csv','macro_metrics.csv','aps_by_task.csv','aps_by_property_macro.csv','metrics.json','compatibility_examples.json','README.md']
    provenance=dict(snapshots=snapshots,endpoint_source_hashes=audit['source_files'],manifest_hashes=manifests,reference_sha256=sha(REFERENCE),sascorer_sha256=sha(ROOT/'data/props/sascorer.py'),sas_fragments_sha256=sha(ROOT/'data/props/fpscores.pkl.gz'),script_sha256=sha(__file__))
    result=dict(status='PASS',completed_utc=datetime.now(timezone.utc).isoformat(),environment=environment,final_k=1,endpoints=70290,configurations=18,task_groups=180,per_configuration_inputs=3905,successes=counts['success'],SR_mismatch_groups=sr_mismatches,endpoint_changes=0,new_generation=0,new_target_property_predictions=0,statistics=dict(counts),provenance=provenance,per_input_uncompressed_sha256=raw_sha.hexdigest(),output_sha256={name:sha(OUT/name) for name in outputs})
    save('audit.json',result);print(json.dumps(dict(status='PASS',endpoints=70290,statistics=dict(counts)),indent=2),flush=True)

if __name__=='__main__': main()
