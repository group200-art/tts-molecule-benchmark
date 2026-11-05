"""Skip an unnecessary property-support probe on already parse-invalid inputs.

This changes no validity, property value, missing-data status or success label.
The original producer already computes no properties when ADMET's authoritative
inf parse is invalid. Its unconditional eval-side RemoveHs probe could nevertheless
raise first. Keep every original expression except that extra probe's guard.
"""
from common import *
import types


def install(module):
    if getattr(module, '_supp_invalid_probe_guard', False):
        return
    path = G / 'standard_producer.py'
    source = path.read_text()
    old = 'if valid and property_error is None:property_error=property_support_error(mol,Chem)'
    new = "if valid and ar['valid'] and property_error is None:property_error=property_support_error(mol,Chem)"
    assert source.count(old) == 1
    tree = ast.parse(source.replace(old, new))
    node = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'produce')
    # The same original namespace, helpers, evaluator, precision and response schema.
    body = ast.Module(body=[node])
    if sys.version_info >= (3, 8):
        body.type_ignores = []
    exec(compile(body, str(path) + ':supp_invalid_probe_guard', 'exec'), module.__dict__)
    module._supp_invalid_probe_guard = True
    module._supp_guard_receipt = dict(original_path=str(path), original_sha256=sha(path),
                                     old=old, new=new,
                                     effect='Skip property-support probe only when original authoritative parse validity is false; all values remain missing and original success is false.')


def install_got(module):
    cls=module.EvalService
    if getattr(cls,'_supp_invalid_probe_guard',False):return
    path=G/'code/property_service.py';source=path.read_text()
    old='property_error = property_support_error(mol,self.Chem) if valid else None'
    new="property_error = property_support_error(mol,self.Chem) if valid and req['rdkit_valid'] else None"
    assert source.count(old)==1
    tree=ast.parse(source.replace(old,new));klass=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='EvalService');node=next(x for x in klass.body if isinstance(x,ast.FunctionDef) and x.name=='handle')
    body=ast.Module(body=[node])
    if sys.version_info>=(3,8):body.type_ignores=[]
    namespace=dict(module.__dict__);exec(compile(body,str(path)+':supp_invalid_probe_guard','exec'),namespace);cls.handle=namespace['handle'];cls._supp_invalid_probe_guard=True
    cls._supp_guard_receipt=dict(original_path=str(path),original_sha256=sha(path),old=old,new=new,effect='Skip only unnecessary property probe for authoritative parse-invalid requests; scorer and numeric evaluators unchanged.')
