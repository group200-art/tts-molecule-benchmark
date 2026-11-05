"""Post-generation raw-SMILES similarity; never changes the online GoT scorer."""
from common import *
from functools import lru_cache
import argparse

def main():
 require_allocation()
 assert sys.version_info[:2]==(3,6)
 from rdkit import rdBase,Chem
 assert rdBase.rdkitVersion=='2020.03.3'
 sys.path.insert(0,str(REPO/'data/props'))
 import properties
 @lru_cache(maxsize=20000)
 def similarity(source,candidate):
  for raw in [source,candidate]:
   mol=Chem.MolFromSmiles(raw)
   if mol is None or mol.GetNumAtoms()==0:raise ValueError('Successful source/candidate cannot be parsed for full-precision similarity')
  return properties.similarity(source,candidate)
 parser=argparse.ArgumentParser();parser.add_argument('--unit',required=True);args=parser.parse_args()
 unit=Path(args.unit).resolve();assert unit.parent==OUT/'generation'
 completed=0;differences=0
 for path in sorted((unit/'completed').glob('*/*.json')):
  dest=unit/'readout_similarity'/path.parent.name/path.name;source_hash=sha(path)
  if dest.exists():
   old=read(dest);assert old['source_sha256']==source_hash
   completed+=1;continue
  started=time.perf_counter();before=similarity.cache_info();row=read(path);values={};ncomparisons=0
  for arm in row['arms']:
   cs=arm.get('pool') or [t['candidate'] for lane in arm['trajectories'] for t in lane['trials']]
   for c in cs:
    if not labels(row['record'],c)['success']:continue
    rid=c['response']['record_id']
    if rid not in values:
     value=similarity(row['record']['source_smiles'],c['raw_smiles'])
     if not isinstance(value,(float,int)) or not math.isfinite(value):raise RuntimeError('Successful candidate has no finite raw-SMILES similarity: '+rid)
     values[rid]=value
     existing=(c.get('raw_properties') or {}).get('sim')
     if existing is not None:
      ncomparisons+=1
      if existing!=value:differences+=1
  write(dest,dict(status='COMPLETE',source_path=str(path),source_sha256=source_hash,source_smiles=row['record']['source_smiles'],rdkit=rdBase.rdkitVersion,similarity_function_sha256=sha(REPO/'data/props/properties.py'),similarity_full_precision_by_response_id=values,successful_distinct_records=len(values),compared_with_online_similarity=ncomparisons,access='offline; no influence on generation, scorer, ordering or stopping',actual_similarity_function_calls=similarity.cache_info().misses-before.misses,similarity_cache_hits=similarity.cache_info().hits-before.hits,wall_seconds=time.perf_counter()-started))
  completed+=1
 write(unit/'similarity_completion.json',dict(status='COMPLETE',paired_input_files=completed,comparison_differences_in_newly_computed_records=differences,comparison_is_diagnostic_only=True))
 log('generation_raw_similarity_complete',unit=unit.name,paired_inputs=completed)
if __name__=='__main__':main()
