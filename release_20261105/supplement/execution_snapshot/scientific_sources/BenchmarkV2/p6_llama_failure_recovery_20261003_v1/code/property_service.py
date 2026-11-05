"""Persistent, CPU-only property RPC; exact request joins, no cross-graph answer cache."""
import contextlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from common import ROOT, REPO, INF_PY, EVAL_PY, M1, M2, append, write, read, digest, sha, require_allocation, task_contract
from precision import finite_or_none, rounded_csv
from property_error_isolation import property_support_error


class InfService:
    def __init__(self, task_id, directory):
        from rdkit import Chem, rdBase
        from importlib.metadata import version
        self.Chem, self.model, self.directory = Chem, None, directory
        self.columns = read(ROOT / 'sources/properties.json')['tasks'][task_id]['admet_columns']
        self.identity = {'environment': 'gellmo_inf', 'python': platform.python_version(), 'rdkit': rdBase.rdkitVersion,
                         'admet_requirement': 'required' if self.columns else 'not_required', 'device': 'cpu'}
        self.identity['package_versions_without_import'] = {p: version(p) for p in ['torch', 'pandas', 'numpy', 'admet-ai', 'chemprop']}
        if rdBase.rdkitVersion != '2024.03.6':
            raise RuntimeError('inf RDKit version differs from pinned environment: ' + rdBase.rdkitVersion)
        self.cost = {'model_load_calls': 0, 'load_wall_seconds': 0.0, 'predict_started': 0,
                     'predict_completed': 0, 'predict_wall_seconds': 0.0, 'admet_imported': False, 'shared_setup_wall_seconds': 0.0}

    def handle(self, data):
        op = data['op']
        if op == 'identity':
            return self.identity
        if op == 'recheck':
            rows = []
            for node in data['nodes']:
                raw = node['parsed_smiles']
                mol = self.Chem.MolFromSmiles(raw) if raw.strip() else None
                valid = mol is not None and mol.GetNumAtoms() > 0
                canonical = self.Chem.MolToSmiles(mol) if valid else ''
                roundtrip = self.Chem.MolFromSmiles(canonical) if valid and self.columns else None
                roundtrip_valid = bool(roundtrip is not None and roundtrip.GetNumAtoms() > 0) if valid and self.columns else None
                fallback = bool(valid and self.columns and not roundtrip_valid)
                request_smiles = raw if fallback or not valid else canonical
                request_mol = self.Chem.MolFromSmiles(request_smiles) if valid and self.columns else None
                property_error = property_support_error(request_mol,self.Chem) if request_mol is not None else None
                rows.append({'node_id': node['node_id'], 'parsed_smiles': raw, 'rdkit_valid': bool(valid),
                             'inf_canonical': canonical, 'inf_rdkit_version': self.identity['rdkit'],
                             'canonical_roundtrip_valid': roundtrip_valid,
                             'request_smiles': request_smiles, 'property_error': property_error,
                             'request_representation': 'parsed_fallback' if fallback else ('canonical' if valid else 'invalid_parsed'),
                             'fallback_reason': 'inf_valid_raw_canonical_roundtrip_failed' if fallback else None,
                             'preprocessing_version': M2})
            return rows
        assert op == 'predict'
        strings = data['strings']
        assert len(strings) == len(set(strings)) and len(strings) <= 100
        if not self.columns or not strings:
            return {'by_smiles': {s: {} for s in strings}, 'cost': self.cost, 'identity': self.identity}
        if self.model is None:
            setup_started = time.perf_counter()
            import torch
            import pandas as pd
            from admet_ai import ADMETModel
            from admet_ai.constants import DEFAULT_MODELS_DIR
            from importlib.metadata import version
            assert version('admet-ai') == '1.3.1'
            assert not torch.cuda.is_available()
            torch.set_num_threads(8)
            self.identity.update(torch=torch.__version__, pandas=pd.__version__, admet_ai=version('admet-ai'),
                                 model_directory=str(DEFAULT_MODELS_DIR), num_workers=0, fingerprint_multiprocessing_min=101)
            self.cost.update(model_load_calls=1, admet_imported=True)
            write(self.directory / 'inf_load_started.json', {'identity': self.identity, 'cost': self.cost})
            t = time.perf_counter()
            original_torch_load = torch.load
            self.cost['internal_torch_load_events'] = []
            def measured_torch_load(*args, **kwargs):
                event = {'path': str(args[0] if args else kwargs.get('f')), 'started': True, 'completed': False}
                self.cost['internal_torch_load_events'].append(event)
                started = time.perf_counter()
                try:
                    answer = original_torch_load(*args, **kwargs)
                    event['completed'] = True
                    return answer
                finally:
                    event['wall_seconds'] = time.perf_counter() - started
            torch.load = measured_torch_load
            try:
                self.model = ADMETModel(models_dir=DEFAULT_MODELS_DIR, num_workers=0, fingerprint_multiprocessing_min=101)
                assert str(self.model.device) == 'cpu'
                self.cost['internal_ensemble_model_counts'] = [len(models) for models in self.model.model_lists]
                self.cost['internal_weight_models_loaded'] = sum(self.cost['internal_ensemble_model_counts'])
            finally:
                torch.load = original_torch_load
                self.cost['load_wall_seconds'] = time.perf_counter() - t
                self.cost['shared_setup_wall_seconds'] = time.perf_counter() - setup_started
                write(self.directory / 'inf_load_result.json', {'identity': self.identity, 'cost': self.cost, 'completed': self.model is not None})
        self.cost['predict_started'] += 1
        append(self.directory / 'inf_predictions.jsonl', dict(event='started', context=data['context'], strings=strings))
        t = time.perf_counter()
        result = None
        try:
            import pandas as pd
            frame = self.model.predict(smiles=strings)
            assert isinstance(frame, pd.DataFrame) and list(frame.index) == strings
            assert all(c in frame.columns for c in self.columns.values())
            result = {s: {n: finite_or_none(frame.loc[s, col]) for n, col in self.columns.items()} for s in strings}
            self.cost['predict_completed'] += 1
            return {'by_smiles': result, 'cost': self.cost, 'identity': self.identity}
        finally:
            elapsed = time.perf_counter() - t
            self.cost['predict_wall_seconds'] += elapsed
            append(self.directory / 'inf_predictions.jsonl', dict(event='result', context=data['context'], strings=strings,
                          columns=self.columns, raw_targets=result, completed=result is not None,
                          predict_wall_seconds=elapsed, cumulative=self.cost))


class EvalService:
    def __init__(self, task_id, directory):
        from rdkit import Chem, rdBase
        import pandas as pd
        import numpy as np
        assert sys.version_info[:2] == (3, 6) and pd.__version__ == '1.0.5' and np.__version__ == '1.18.5'
        self.Chem, self.directory = Chem, directory
        sys.path.insert(0, str(REPO / 'data/props'))
        import properties
        self.properties = properties
        self.shared_load_seconds = 0.0
        for module, name in [(properties.drd2_scorer, 'load_model'), (properties.sascorer, 'readFragmentScores')]:
            original = getattr(module, name)
            def measured_loader(*args, _original=original, _name=name, **kwargs):
                t = time.perf_counter()
                completed = False
                try:
                    answer = _original(*args, **kwargs)
                    completed = True
                    return answer
                finally:
                    elapsed = time.perf_counter() - t
                    self.shared_load_seconds += elapsed
                    append(self.directory / 'standard_asset_loads.jsonl', {'loader': _name, 'completed': completed, 'wall_seconds': elapsed})
            setattr(module, name, measured_loader)
        assert sha(properties.__file__) == '771722423bdb76955724e6d8a3c8298cedf29c55cf631b847640b4dfc6bec812'
        self.names = [o['property'] for o in task_contract(task_id)['objectives']]
        self.identity = {'environment': 'gellmo_eval_py36', 'python': platform.python_version(), 'rdkit': rdBase.rdkitVersion,
                         'pandas': pd.__version__, 'numpy': np.__version__, 'properties_sha256': sha(properties.__file__), 'device': 'cpu'}

    def handle(self, data):
        if data['op'] == 'identity':
            return self.identity
        if data['op'] == 'validity_only':
            checked = []
            for raw in data['strings']:
                mol = self.Chem.MolFromSmiles(raw) if raw.strip() else None
                checked.append({'request_smiles': raw, 'eval_valid': bool(mol is not None and mol.GetNumAtoms() > 0)})
            return checked
        assert data['op'] == 'evaluate'
        rows = []
        for req in data['requests']:
            ident = req['identity']
            raw = ident['request_smiles']
            mol = self.Chem.MolFromSmiles(raw) if raw.strip() else None
            valid = mol is not None and mol.GetNumAtoms() > 0
            vals = None
            t = time.perf_counter()
            shared_before = self.shared_load_seconds
            property_error = property_support_error(mol,self.Chem) if valid else None
            property_error = req.get('property_error') or property_error
            if valid and req['rdkit_valid'] and property_error is None:
                vals = dict(req['admet_raw_targets'])
                if 'drd2' in self.names:
                    vals['drd2'] = finite_or_none(self.properties.drd2(raw))
                if 'plogp' in self.names:
                    vals['plogp'] = finite_or_none(self.properties.penalized_logp(raw))
                if 'qed' in self.names:
                    vals['qed'] = finite_or_none(self.properties.qed(raw))
                vals['sas'] = finite_or_none(self.properties.sas(raw))
                vals['sim'] = finite_or_none(self.properties.similarity(ident['source_smiles'], raw))
            comparison, serialized, versions = rounded_csv(vals) if vals is not None else (None, None, None)
            row = {'request_id': req['request_id'], 'request_smiles': raw, 'identity': ident,
                   'rdkit_valid': req['rdkit_valid'], 'eval_valid': bool(valid), 'eval_canonical': self.Chem.MolToSmiles(mol) if valid else '',
                   'eval_rdkit_version': self.identity['rdkit'], 'inf_canonical': req['inf_canonical'],
                   'environment_validity_disagreement': bool(valid) != req['rdkit_valid'],
                   'property_error': property_error,
                   'raw_properties': vals, 'properties': comparison, 'property_row_exists': vals is not None,
                   'rounding_csv': serialized, 'rounding_versions': versions,
                   'standard_property_and_precision_seconds': time.perf_counter() - t,
                   'shared_standard_load_seconds': self.shared_load_seconds - shared_before,
                   'nonfinite_or_missing_fields': [p for p in self.names + ['sas', 'sim'] if finite_or_none((comparison or {}).get(p)) is None]}
            rows.append(row)
            append(self.directory / 'eval_tool_rows.jsonl', {'context': data.get('context'), 'row': row})
        return rows


def service_main(kind, task_id, directory):
    require_allocation()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise PermissionError('Property subprocess must hide GPUs before imports')
    protocol_out = sys.stdout
    service = None
    for line in sys.stdin:
        try:
            data = json.loads(line)
            if data['op'] == 'stop':
                protocol_out.write(json.dumps({'ok': True, 'result': {'stopped': True}}) + '\n')
                protocol_out.flush()
                return
            with contextlib.redirect_stdout(sys.stderr):
                if service is None:
                    service = (InfService if kind == 'inf' else EvalService)(task_id, directory)
                result = service.handle(data)
            protocol_out.write(json.dumps({'ok': True, 'result': result}, allow_nan=False) + '\n')
            protocol_out.flush()
        except BaseException as err:
            payload = {'ok': False, 'exception': repr(err), 'traceback': traceback.format_exc()}
            append(directory / (kind + '_failures.jsonl'), payload)
            protocol_out.write(json.dumps(payload) + '\n')
            protocol_out.flush()
            raise


class RPC:
    def __init__(self, kind, task_id, directory):
        self.kind, self.directory = kind, directory
        self.stderr = (directory / (kind + '_stderr.log')).open('x')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
        py = INF_PY if kind == 'inf' else EVAL_PY
        self.start_time = time.perf_counter()
        self.process = subprocess.Popen([py, '-B', '-u', str(ROOT / 'code/property_service.py'), kind, task_id, str(directory)],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr, universal_newlines=True, bufsize=1, env=env, cwd=str(ROOT))
        append(directory / 'property_processes.jsonl', {'event': 'started', 'kind': kind, 'pid': self.process.pid, 'interpreter': py})

    def request(self, data):
        if self.process.poll() is not None:
            raise RuntimeError('Property process already terminated: ' + self.kind)
        t = time.perf_counter()
        self.process.stdin.write(json.dumps(data, ensure_ascii=False, allow_nan=False) + '\n')
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('Property process closed RPC without response: ' + self.kind)
        response = json.loads(line)
        append(self.directory / 'property_rpc.jsonl', {'kind': self.kind, 'op': data['op'], 'context': data.get('context'),
                   'wall_seconds': time.perf_counter() - t, 'ok': response['ok']})
        if not response['ok']:
            raise RuntimeError(self.kind + ': ' + response['exception'])
        return response['result']

    def close(self):
        if self.process.poll() is None:
            self.request({'op': 'stop'})
            self.process.stdin.close()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=10)
        append(self.directory / 'property_processes.jsonl', {'event': 'terminated', 'kind': self.kind, 'pid': self.process.pid,
                        'returncode': self.process.returncode, 'lifetime_wall_seconds': time.perf_counter() - self.start_time})
        self.stderr.close()


class PropertyBridge:
    def __init__(self, task_id, directory):
        self.directory, self.task_id = directory, task_id
        self.services = {}
        self.identity = {}
        self.graph_cache = {}
        self.cost = {'by_G': {}, 'startup_seconds': 0.0}
        self.last_shared_load_seconds = 0.0

    def start(self):
        t = time.perf_counter()
        self.bridge_code_identity = {'property_service.py': sha(ROOT / 'code/property_service.py'), 'precision.py': sha(ROOT / 'code/precision.py')}
        try:
            for kind in ('inf', 'eval'):
                self.services[kind] = RPC(kind, self.task_id, self.directory)
                self.identity[kind] = self.services[kind].request({'op': 'identity'})
            write(self.directory / 'property_identity.json', self.identity)
        finally:
            self.cost['startup_seconds'] = time.perf_counter() - t
            write(self.directory / 'property_cost.json', self.cost, replace=True)

    def begin_graph(self, record, G):
        self.record, self.G = record, G
        self.graph_cache = {}
        self.cost['by_G'].setdefault(str(G), {'stage_rpc_seconds_inclusive': 0.0, 'stage_seconds_excluding_shared_load': 0.0,
                                           'requests_new': 0, 'requests_cached': 0})

    def fill(self, nodes, property_map, stage):
        newnodes = [n for n in nodes if n['node_id'] not in property_map]
        assert len(newnodes) == (self.G if stage == 'wave0' else 16)
        context = {'task_id': self.task_id, 'input_uid': self.record['input_uid'], 'G': self.G, 'stage': stage}
        t = time.perf_counter()
        checked = self.services['inf'].request(dict(op='recheck', nodes=newnodes, context=context))
        assert [r['node_id'] for r in checked] == [n['node_id'] for n in newnodes]
        contracts = read(ROOT / 'sources/properties.json')['tasks'][self.task_id]['cache_contract']
        contract = dict(contracts, preprocessing=M2, validity_gate=M1, environment_versions=self.identity,
                        bridge_code_sha256=self.bridge_code_identity)
        fallback_strings = list(dict.fromkeys(r['request_smiles'] for r in checked if r['request_representation'] == 'parsed_fallback'))
        if fallback_strings:
            validated = self.services['eval'].request(dict(op='validity_only', strings=fallback_strings, context=context))
            assert [r['request_smiles'] for r in validated] == fallback_strings
            append(self.directory / 'fallback_validity_checks.jsonl', {'context': context, 'checked': checked, 'eval_checks': validated})
        request_map, mapping = {}, []
        for row in checked:
            raw = row['request_smiles']
            ident = {'task_id': self.task_id, 'source_smiles': self.record['source_smiles'], 'request_smiles': raw,
                     'cache_contract': contract}
            rid = 'p8-property-v1-' + digest(ident)
            mapping.append({'node_id': row['node_id'], 'request_id': rid, 'recheck': row})
            if rid not in self.graph_cache and rid not in request_map:
                request_map[rid] = {'request_id': rid, 'identity': ident, 'rdkit_valid': row['rdkit_valid'],
                                    'inf_canonical': row['inf_canonical'], 'property_error': row.get('property_error')}
        requests = list(request_map.values())
        max_batch = self.G if stage == 'wave0' else 16
        assert len(requests) <= max_batch
        strings = [r['identity']['request_smiles'] for r in requests if r['rdkit_valid'] and not r.get('property_error')]
        assert len(strings) == len(set(strings))
        admet = self.services['inf'].request(dict(op='predict', strings=strings, context=context))
        assert set(admet['by_smiles']) == set(strings)
        for req in requests:
            req['admet_raw_targets'] = admet['by_smiles'].get(req['identity']['request_smiles'], {})
        append(self.directory / 'property_requests.jsonl', {'context': context, 'requests': requests, 'mapping': mapping})
        responses = self.services['eval'].request(dict(op='evaluate', requests=requests, context=context))
        assert [r['request_id'] for r in responses] == [r['request_id'] for r in requests]
        for req, response in zip(requests, responses):
            assert response['identity'] == req['identity'] and response['request_smiles'] == req['identity']['request_smiles']
            assert response['request_id'] not in self.graph_cache
            self.graph_cache[response['request_id']] = response
        append(self.directory / 'property_responses.jsonl', {'context': context, 'responses': responses, 'admet_identity': admet['identity']})
        for m in mapping:
            r = dict(self.graph_cache[m['request_id']])
            r['inf_rdkit_version'] = m['recheck']['inf_rdkit_version']
            r['parsed_smiles'] = m['recheck']['parsed_smiles']
            r['node_id'] = m['node_id']
            for key in ('canonical_roundtrip_valid', 'request_representation', 'fallback_reason', 'preprocessing_version'):
                r[key] = m['recheck'][key]
            r['validity_gate_version'] = M1
            property_map[m['node_id']] = r
        cost = self.cost['by_G'][str(self.G)]
        elapsed = time.perf_counter() - t
        shared = admet['cost']['shared_setup_wall_seconds']
        standard_shared = sum(r['shared_standard_load_seconds'] for r in responses)
        cost['stage_rpc_seconds_inclusive'] += elapsed
        cost['stage_seconds_excluding_shared_load'] += elapsed - (shared - self.last_shared_load_seconds) - standard_shared
        self.cost['standard_shared_load_seconds'] = self.cost.get('standard_shared_load_seconds', 0.0) + standard_shared
        self.last_shared_load_seconds = shared
        cost['requests_new'] += len(requests)
        cost['requests_cached'] += len(mapping) - len(requests)
        self.cost['admet_shared_and_nested_cost'] = admet['cost']
        write(self.directory / 'property_cost.json', self.cost, replace=True)

    def close(self):
        errors = []
        for kind, service in self.services.items():
            try:
                service.close()
            except BaseException as err:
                errors.append({'kind': kind, 'error': repr(err)})
        write(self.directory / 'property_cost.json', dict(self.cost, close_errors=errors), replace=True)
        if errors:
            raise RuntimeError('Property subprocess termination failure: ' + repr(errors))


if __name__ == '__main__':
    from pathlib import Path
    service_main(sys.argv[1], sys.argv[2], Path(sys.argv[3]))
