"""Generation only. One frozen four-input/19-trajectory batch, four independent controller steps."""
import argparse
import fcntl
import importlib
import time
import traceback
from common import *


class Backend:
    def __init__(self, contract, directory, attempt):
        self.contract, self.directory, self.attempt = contract, directory, attempt
        self.loaded = False

    def start(self):
        # Same audited BF16 unmerged PEFT assembly/tokenizer used by completed Llama ICL.
        # Its ToT prompts and generation method are never used.
        sys.path.insert(0, str(K1/'code'))
        tp = importlib.import_module('tot_protocol')
        tp.ROOT = ROOT
        self.tb = importlib.import_module('tot_backend')
        self.assembly = self.tb.Backend('llama', self.contract['model_assembly'],
                                        sha(ROOT/'contract.json'), self.attempt)
        self.assembly.start()
        import torch
        gpu = torch.cuda.get_device_properties(0)
        assert 'RTX PRO 6000 Blackwell Server Edition' in gpu.name and gpu.total_memory > 90*1024**3
        self.loaded = True

    def generate(self, role, r, prompts, batch):
        path = self.directory/'calls'/batch['batch_id']/('{}_{}.json.gz'.format(role, r))
        ident = call_identity(batch, role, r, prompts)
        if path.exists():
            old = read(path)
            assert old['identity'] == ident and old['prompts'] == prompts and old['status'] == 'complete'
            assert len(old['outputs']) == len(prompts)
            return old['outputs'], old['measurement']
        if not self.loaded:
            self.start()
        from transformers import GenerationConfig
        import torch
        tokenizer, model = self.assembly.tok, self.assembly.model
        encoded = [tokenizer.encode(p, add_special_tokens=True, truncation=False) for p in prompts]
        cap = self.contract['role_caps'][role]
        assert all(x and x[0] == 128000 and len(x) <= self.contract['prompt_limit'] and
                   len(x)+cap <= self.contract['context_limit'] for x in encoded), 'Context overflow; no truncation'
        width = max(map(len, encoded))
        ids = torch.tensor([[128009]*(width-len(x))+x for x in encoded], dtype=torch.long, device='cuda:0')
        mask = torch.tensor([[0]*(width-len(x))+[1]*len(x) for x in encoded], dtype=torch.long, device='cuda:0')
        config = GenerationConfig.from_dict(self.assembly.defaults)
        assert not config.update(**dict(self.contract['decoding'], max_new_tokens=cap,
            bos_token_id=128000, eos_token_id=128009, pad_token_id=128009, return_dict_in_generate=False))
        config.validate()
        started_path = path.with_name(path.name+'.started.json')
        if started_path.exists():
            assert read(started_path)['identity'] == ident
        else:
            save(started_path, dict(identity=ident, effective_decoding=config.to_dict(),
                                   started_unix=time.time(), job=os.environ['SLURM_JOB_ID']))
        self.tb.seed_all(ident['seed'])
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            sequences = model.generate(input_ids=ids, attention_mask=mask,
                                       generation_config=config, synced_gpus=False).detach().cpu().tolist()
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        assert len(sequences) == len(prompts)
        outputs, padded = [], ids.cpu().tolist()
        for row, (tokens, (record, seed)) in enumerate(zip(sequences, ordered_lanes(batch))):
            assert tokens[:width] == padded[row]
            continuation = tokens[width:]
            assert 1 <= len(continuation) <= cap
            generated, content, ended = self.tb.trim_tokens(continuation, 128009)
            # Original SR decoding used the full padded continuation, skip_special_tokens=True.
            raw = tokenizer.decode(continuation, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            outputs.append(dict(raw_text=raw, generated_ids=generated, content_ids=content,
                input_tokens=len(encoded[row]), output_tokens=len(generated), ended_with_eos=ended,
                post_eos_padding_tokens=len(continuation)-len(generated), role=role, r=r,
                seed=ident['seed'], rep_seed=seed, batch_row=row, batch_id=batch['batch_id'],
                call_index=ROLES.index((role,r)), prompt_digest=ident['prompt_digests'][row],
                call_path=str(path), max_new_tokens=cap,
                reached_cap_without_eos=not ended and len(generated)==cap))
        measurement = dict(model_generate_seconds=elapsed, role=role, r=r, physical_batch_rows=len(outputs),
            input_tokens=sum(len(x) for x in encoded), output_tokens=sum(x['output_tokens'] for x in outputs),
            max_input_tokens=width, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved())
        save(path, dict(status='complete', identity=ident, prompts=prompts, input_ids=encoded,
                        effective_decoding=config.to_dict(), outputs=outputs, measurement=measurement,
                        slurm_job_id=os.environ['SLURM_JOB_ID']))
        print(json.dumps(dict(event='call_complete', batch=batch['batch_id'], **measurement)), flush=True)
        return outputs, measurement


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--unit-index', type=int)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    contract = verify()
    units = read(ROOT/'units.json')
    if args.audit_only:
        assert sum(x['n_inputs'] for x in units) == 3905
        print(json.dumps(dict(status='CPU_IDENTITIES_VALID', new_terminals=74195)))
        return
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('LLAMA_SR_EXECUTION') == EXECUTION
    assert os.environ.get('SLURM_RESTART_COUNT','0') == '0'
    if args.probe:
        chosen = [units[i] for i in contract['probe_unit_indices']]
    else:
        gate = read(ROOT/'reports/engineering_gate.json')
        assert gate['status'] == 'PASS' and gate['freeze_sha256'] == sha(ROOT/'reports/freeze.json')
        assert gate['contract_sha256'] == sha(ROOT/'contract.json')
        assert args.unit_index is not None and 0 <= args.unit_index < len(units)
        chosen = [units[args.unit_index]]
    attempt = ROOT/'attempts'/(os.environ['SLURM_JOB_ID']+'_'+str(os.getpid()))
    attempt.mkdir(parents=True, exist_ok=False)
    templates = read(contract['templates_path'])
    backend = Backend(contract, ROOT/'generation', attempt)
    completed, status = 0, dict(status='running', probe=args.probe, slurm_job_id=os.environ['SLURM_JOB_ID'])
    try:
        for unit in chosen:
            directory = ROOT/'generation'/'units'/unit['unit_id']
            directory.mkdir(parents=True, exist_ok=True)
            with (directory/'.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
                for batch in unit['batches'][:1] if args.probe else unit['batches']:
                    dest = directory/(batch['batch_id']+'.json.gz')
                    if dest.exists():
                        result = read(dest)
                        audit_batch(result, batch, templates)
                    else:
                        result = execute_batch(batch, backend.generate, templates)
                        save(dest, result)
                    completed += len(result['lanes'])
                    print(json.dumps(dict(event='batch_complete', batch=batch['batch_id'], new_terminals=len(result['lanes']))), flush=True)
                if not args.probe:
                    paths = [directory/(b['batch_id']+'.json.gz') for b in unit['batches']]
                    assert sum(len(read(p)['lanes']) for p in paths) == unit['n_inputs']*19
                    save(directory/'complete.json', dict(status='GENERATION_COMPLETE_PROPERTIES_PENDING',
                        contract_sha256=sha(ROOT/'contract.json'), unit=unit,
                        paths=[dict(path=str(p),sha256=sha(p)) for p in paths], terminals=unit['n_inputs']*19))
        assert not any(x == 'rdkit' or x.startswith('rdkit.') or x == 'admet_ai' for x in sys.modules)
        status.update(status='PROBE_GENERATION_COMPLETE' if args.probe else 'UNIT_GENERATION_COMPLETE', new_terminals=completed)
    except BaseException as error:
        status.update(status='FAILED_NO_REFILL', error=repr(error), traceback=traceback.format_exc(), new_terminals=completed)
        raise
    finally:
        save(attempt/'status.json', status)


if __name__ == '__main__':
    main()
