"""Synchronous local IPC, immutable requests and explicit infrastructure failures."""
import subprocess
import time
from common import *
from property_bridge import requests_for


class PropertyClient:
    def __init__(self, directory):
        self.directory = directory
        self.processes = {}
        self.logs = []

    def invoke(self, mode, command):
        done = Path(command['done'])
        if not done.exists():
            if mode not in self.processes:
                envname = 'gellmo_inf' if mode == 'inf' else 'gellmo_eval_py36'
                log = (self.directory / (mode + '.log')).open('ab')
                self.logs.append(log)
                self.processes[mode] = subprocess.Popen([ENVS[envname], '-B', str(ROOT / 'property_service.py'), mode],
                    stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, universal_newlines=True,
                    env=isolate(self.directory / ('env_' + mode), cpu=True))
            proc = self.processes[mode]
            proc.stdin.write(json.dumps(command) + '\n')
            proc.stdin.flush()
            start = time.monotonic()
            while not done.exists():
                if proc.poll() is not None:
                    raise RuntimeError('Property service exited: ' + mode + '; see ' + str(self.directory))
                if time.monotonic() - start > 3600:
                    raise TimeoutError('Property request exceeded one hour')
                time.sleep(0.1)
        result = read(done)
        if result['status'] != 'complete':
            raise RuntimeError(result.get('error', repr(result)))
        return result

    def evaluate(self, method, task, records, candidates, kind, directory, cache_check=False):
        if (directory / 'complete.json').exists():
            result = read(directory / 'complete.json')
            bundle = read(directory / 'requests.json')
            assert bundle['candidates'] == candidates and bundle['records'] == records and bundle['kind'] == kind
            return read(result['result_path'])
        directory.mkdir(parents=True, exist_ok=True)
        spec = property_spec(method, task)
        requests, mapping = requests_for(candidates, spec['cache_contract'])
        data = dict(method=method, task_id=task, records=records, candidates=candidates, kind=kind,
                    requests=requests, mapping=mapping, **spec)
        path = directory / 'requests.json'
        if path.exists():
            assert read(path) == data
        else:
            save(path, data)
        # A failed incomplete attempt is retained; only uncommitted CPU work is retried.
        attempt = directory / ('attempt_' + os.environ['SLURM_JOB_ID'] + '_' + str(os.getpid()))
        attempt.mkdir(exist_ok=False)
        inf = self.invoke('inf', dict(requests=str(path), output=str(attempt / 'inf'), done=str(attempt / 'inf.done.json')))
        standard = self.invoke('standard', dict(requests=str(path), admet=inf['result']['result_path'],
                    output=str(attempt / 'standard'), done=str(attempt / 'standard.done.json')))
        if cache_check:
            repeat = self.invoke('inf', dict(requests=str(path), output=str(attempt / 'cache_check'),
                                            done=str(attempt / 'cache_check.done.json')))
            first = read(inf['result']['result_path'])
            second = read(repeat['result']['result_path'])
            assert first['rows'] == second['rows'], 'Persistent predictor changed outputs on identical batch'
            assert repeat['result']['actual_model_load_calls'] == 0
            save(directory / 'cache_equivalence.json', dict(status='PASS', identical_rows=True,
                 n=len(first['rows']), first=inf['result']['result_path'], cached=repeat['result']['result_path']))
        result = dict(status='complete', result_path=standard['result']['result_path'],
                      requests_sha256=sha(path), inf=inf, standard=standard, plan_sha256=sha(ROOT / 'plan.json'))
        save(directory / 'complete.json', result)
        print(json.dumps(dict(event='properties', kind=kind, candidates=len(candidates),
                              seconds=round(inf['wall_seconds'] + standard['wall_seconds'], 2))), flush=True)
        return read(result['result_path'])

    def close(self):
        for proc in self.processes.values():
            proc.stdin.close()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait(timeout=15)
        for log in self.logs:
            log.close()
