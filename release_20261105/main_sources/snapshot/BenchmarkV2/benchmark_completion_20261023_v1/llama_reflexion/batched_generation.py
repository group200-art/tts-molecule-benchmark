"""The validated Llama sampling kernel, bound to Reflexion's ordered requests."""
from common import *
from replay import group_seed


class BatchedGeneration:
    def __init__(self, method, directory):
        assert method == 'reflexion'
        self.directory = directory
        self.model = None

    def start(self):
        module = load_module('frozen_llama_sampling', S1 / 'runner.py')
        module.ROOT = ROOT
        module.group_seed = group_seed

        def scoped_save(path, value):
            if 'outputs' in value and 'physical_cuda_wall_seconds' in value:
                value['plan_sha256'] = sha(ROOT / 'plan.json')
                value['sampling_adapter_sha256'] = sha(__file__)
            save(path, value)

        module.save = scoped_save
        self.backend = module.Model(plan(), self.directory)
        self.backend.start()
        self.model, self.tok = self.backend.model, self.backend.tok
        self.load_seconds = self.backend.backend.load_seconds

    def run(self, batch, requests, index, path):
        assert all(r['cap'] == 250 and r['minimum'] == 0 and
                   r['role'] in ('thought', 'action', 'reflection') for r in requests)
        return self.backend.run(batch, requests, index, path)
