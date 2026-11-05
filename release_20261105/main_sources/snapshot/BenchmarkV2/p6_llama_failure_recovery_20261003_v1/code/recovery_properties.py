"""Reuse exact recorded GoT property rows, with identity-checked graph joins."""
import json
from property_service import PropertyBridge as OriginalBridge
from common import M1, append
from recovery_generation import recovery_unit, verify_source


def event_key(event):
    c=event['context']
    return (c['task_id'],c['input_uid'],c['G'],c['stage'])


class PropertyBridge(OriginalBridge):
    def __init__(self, task_id, directory):
        super().__init__(task_id,directory)
        unit=recovery_unit(directory)
        self.old=verify_source(unit)
        self.saved_requests={}
        self.saved_responses={}
        for filename,target in [('property_requests.jsonl',self.saved_requests),('property_responses.jsonl',self.saved_responses)]:
            with (self.old/filename).open() as stream:
                for line in stream:
                    event=json.loads(line);key=event_key(event)
                    assert key not in target
                    target[key]=event
        assert set(self.saved_responses)<=set(self.saved_requests)
        self.reused=set()

    def fill(self,nodes,property_map,stage):
        key=(self.task_id,self.record['input_uid'],self.G,stage)
        if key not in self.saved_responses:
            return super().fill(nodes,property_map,stage)
        assert key not in self.reused
        req=self.saved_requests[key];res=self.saved_responses[key]
        newnodes=[n for n in nodes if n['node_id'] not in property_map]
        mapping=req['mapping']
        assert [n['node_id'] for n in newnodes]==[m['node_id'] for m in mapping]
        assert [n['parsed_smiles'] for n in newnodes]==[m['recheck']['parsed_smiles'] for m in mapping]
        answers={r['request_id']:r for r in res['responses']}
        # A request may have been cached in the previous wave of this same graph.
        for earlier_key,earlier in self.saved_responses.items():
            if earlier_key[:3]==key[:3] and earlier_key in self.reused:
                for r in earlier['responses']:answers.setdefault(r['request_id'],r)
        for m in mapping:
            r=dict(answers[m['request_id']]);ident=r['identity']
            assert ident['source_smiles']==self.record['source_smiles']
            assert r['request_smiles']==m['recheck']['request_smiles']
            r.update(inf_rdkit_version=m['recheck']['inf_rdkit_version'],parsed_smiles=m['recheck']['parsed_smiles'],node_id=m['node_id'])
            for field in ('canonical_roundtrip_valid','request_representation','fallback_reason','preprocessing_version'):
                r[field]=m['recheck'][field]
            r['validity_gate_version']=M1
            property_map[m['node_id']]=r
        self.reused.add(key)
        self.cost['reused_request_rows']=self.cost.get('reused_request_rows',0)+len(res['responses'])
        append(self.directory/'replayed_properties.jsonl',dict(context=req['context'],source=str(self.old),
               nodes=len(mapping),request_rows=len(res['responses']),identities_verified=True,computed_now=False))
