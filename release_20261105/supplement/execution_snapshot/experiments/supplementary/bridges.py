"""Load original scientific implementations without changing their source trees."""
from common import *
import types
from functools import lru_cache
from contextlib import contextmanager

@contextmanager
def original_common():
 original=sys.modules['common']; previous_path=list(sys.path); shim=types.ModuleType('common')
 for k,v in vars(original).items():setattr(shim,k,v)
 shim.ROOT=G;shim.CONTRACT=read(G/'configs/design_contract.json')
 shim.M1='P8-VALIDITY-GATE-INF-AUTHORITATIVE-v1';shim.M2='P8-inf-canonical-roundtrip-guard-v1'
 shim.text_sha=lambda s:hashlib.sha256(s.encode()).hexdigest()
 shim.write=lambda p,o,replace=False:write(p,o)
 sys.path.insert(0,str(G/'src'));import contracts
 shim.task_contract=contracts.task_contract
 sys.modules['common']=shim
 try:yield shim
 finally:
  sys.modules['common']=original
  sys.path[:]=previous_path

@lru_cache(None)
def got_modules():
 with original_common():
  sys.path.insert(0,str(G/'code'))
  scorer=load_module('supp_original_scorer',G/'code/scorer.py');sys.modules['scorer']=scorer
  graph=load_module('supp_original_graph',G/'code/graph_adapter.py')
 return scorer,graph

def got_property_module():
 with original_common():
  sys.path.insert(0,str(G/'code'))
  return load_module('supp_got_property_service',G/'code/property_service.py')

@lru_cache(None)
def ref_modules():
 previous_path=list(sys.path);sys.path.insert(0,str(G/'src'));import contracts,property_bridge
 ref=load_module('supp_original_reflexion',B/'p6_llama_main_ten_tasks_v1/src/reflexion.py');sys.modules['reflexion']=ref
 repair=load_module('supp_original_property_repair',R/'recovery_20261027_v1/property_repair.py')
 svc=types.SimpleNamespace(MODULES={},save=lambda p,o,replace=False:write(p,o),require_allocation=require_allocation)
 repair.install(svc)
 sys.path[:]=previous_path
 return ref,property_bridge,svc
