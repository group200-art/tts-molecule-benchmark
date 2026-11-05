"""Crash-window recovery must reuse a committed physical batch without generation."""
from common import *
from backend import Model
import tempfile
base=CODE/'test_artifacts';base.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(prefix='cache_check_',dir=str(base)) as d:
 model=object.__new__(Model);model.directory=Path(d);model.identity={'fixture':True};model.torch=None
 requests=[dict(max_new_tokens=2,prompt='fixture'+str(i),derived_seed=i) for i in range(2)]
 batch_id=digest(dict(requests=requests,model=model.identity));responses=[dict(request=r,record_id=digest(r),raw_text='fixture',physical_batch_id=batch_id) for r in requests]
 write(model.directory/'batches'/(batch_id+'.json'),dict(responses=responses))
 # Simulate a crash before the first response-cache entry is committed.
 assert model._batch(requests)==responses
 paths=[model.directory/'responses'/(digest(r)+'.json') for r in requests]
 assert all(p.exists() for p in paths)
 paths[1].unlink()
 # Simulate a crash after only one per-response entry is committed.
 assert model._batch(requests)==responses and paths[1].exists()
 assert model._batch(requests)==responses
write(OUT/'cache_checks.json',dict(status='PASS',whole_committed_batch_recovered_with_no_response_entries=True,partial_response_commit_recovered=True,model_called=False,fixture_separate_from_results=True))
log('cache_checks_passed')
