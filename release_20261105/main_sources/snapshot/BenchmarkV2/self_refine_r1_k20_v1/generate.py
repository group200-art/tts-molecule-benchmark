"""One seed across all 3905 inputs; four native-HF calls per batched trajectory."""
import argparse
import fcntl
import os
import time
import traceback
from common import *


class Backend:
    def __init__(self, cfg, directory, attempt):
        self.cfg, self.directory, self.attempt = cfg, directory, attempt
        self.loaded = False

    def start(self):
        import torch, transformers, peft
        from dataclasses import asdict
        from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
        from peft import PeftModel
        assert torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported()
        torch.cuda.set_device(0)
        torch.set_num_threads(8)
        gpu = torch.cuda.get_device_properties(0)
        assert 'RTX PRO 6000 Blackwell Server Edition' in gpu.name and gpu.total_memory > 90 * 1024**3
        spec = self.cfg['runtime']
        self.tok = AutoTokenizer.from_pretrained(spec['base_path'], use_fast=True, local_files_only=True, trust_remote_code=False)
        self.tok.pad_token_id = self.tok.eos_token_id
        self.tok.padding_side = 'left'
        assert self.tok.is_fast and (self.tok.bos_token_id, self.tok.eos_token_id, self.tok.pad_token_id) == (1, 2, 2)
        self.defaults = GenerationConfig.from_pretrained(spec['base_path'], local_files_only=True).to_dict()
        started = time.perf_counter()
        base = AutoModelForCausalLM.from_pretrained(spec['base_path'], torch_dtype=torch.bfloat16,
            device_map={'': 'cuda:0'}, local_files_only=True, trust_remote_code=False, use_safetensors=True)
        self.model = PeftModel.from_pretrained(base, spec['adapter_path'], adapter_name='default', is_trainable=False,
            autocast_adapter_dtype=True, low_cpu_mem_usage=False, ephemeral_gpu_offload=False, local_files_only=True)
        self.model.eval()
        torch.cuda.synchronize()
        status = asdict(self.model.get_model_status())
        assert status['enabled'] is True and status['active_adapters'] == ['default'] and status['merged_adapters'] == []
        assert all(str(p.device) == 'cuda:0' for p in self.model.parameters())
        save(self.attempt / 'model_identity.json', dict(gpu=gpu.name, memory_bytes=gpu.total_memory,
             versions=dict(torch=torch.__version__, transformers=transformers.__version__, peft=peft.__version__),
             runtime=spec, adapter_status=status, padding_side='left', tokenizer_fast=True,
             attention_implementation=self.model.config._attn_implementation, load_seconds=time.perf_counter()-started))
        self.loaded = True

    def generate(self, role, r, prompts, batch):
        path = self.directory / 'calls' / batch['batch_id'] / ('{}_{}.json.gz'.format(role, r))
        ident = dict(batch_id=batch['batch_id'], row_identity_sha256=batch['row_identity_sha256'],
                     role=role, r=r, seed=batch_seed(batch, role, r), prompt_digests=[digest(p) for p in prompts],
                     plan_sha256=sha(ROOT / 'plan.json'))
        if path.exists():
            previous = read(path)
            assert previous['identity'] == ident and previous['prompts'] == prompts
            assert previous['complete'] is True
            return previous['outputs'], previous['measurement']
        if not self.loaded:
            self.start()
        import torch
        from transformers import GenerationConfig, set_seed
        cap = self.cfg['science']['max_new_tokens'][role]
        encoded = self.tok(prompts, add_special_tokens=True, padding=True, truncation=False, return_tensors='pt')
        width = encoded['input_ids'].shape[1]
        lengths = encoded['attention_mask'].sum(-1).tolist()
        assert width + cap <= self.cfg['science']['context_limit']
        config = GenerationConfig.from_dict(self.defaults)
        assert not config.update(**dict(self.cfg['science']['decoding'], max_new_tokens=cap,
            bos_token_id=1, eos_token_id=2, pad_token_id=2, return_dict_in_generate=False))
        config.validate()
        set_seed(ident['seed'])
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            seq = self.model.generate(**{k: v.to('cuda:0') for k, v in encoded.items()},
                                      generation_config=config, synced_gpus=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        output = []
        sequences = seq.detach().cpu().tolist()
        assert len(sequences) == len(prompts)
        for i, ids in enumerate(sequences):
            assert ids[:width] == encoded['input_ids'][i].tolist()
            cont = ids[width:]
            assert 1 <= len(cont) <= cap
            count = cont.index(2) + 1 if 2 in cont else len(cont)
            assert all(t == 2 for t in cont[count:])
            output.append(dict(raw_text=self.tok.decode(cont, skip_special_tokens=True, clean_up_tokenization_spaces=False),
                generated_ids=cont[:count], input_tokens=lengths[i], output_tokens=count,
                reached_cap_without_eos=2 not in cont, seed=ident['seed'], batch_row=i,
                batch_id=batch['batch_id'], call_index=ROLES.index((role, r)), role=role, r=r,
                prompt_digest=ident['prompt_digests'][i], call_path=str(path)))
        measurement = dict(cuda_wall_seconds=elapsed, batch_size=len(prompts), role=role, r=r,
                           input_tokens=sum(lengths), output_tokens=sum(x['output_tokens'] for x in output),
                           max_input_tokens=max(lengths), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                           peak_reserved_bytes=torch.cuda.max_memory_reserved())
        save(path, dict(identity=ident, prompts=prompts, input_ids=encoded['input_ids'].tolist(),
             attention_mask=encoded['attention_mask'].tolist(), effective_generation_config=config.to_dict(),
             outputs=output, measurement=measurement, complete=True, job_id=os.environ['SLURM_JOB_ID']))
        print(json.dumps(dict(event='call_complete', batch=batch['batch_id'], **measurement)), flush=True)
        return output, measurement


def execute_batch(batch, generate, templates):
    lanes = [dict(record=rec, answers=[], feedbacks=[]) for rec in batch['rows']]
    measurements = []
    for role, r in ROLES:
        prompts = [prompt_for(role, task_text(lane['record']), lane['answers'], lane['feedbacks'], r, templates) for lane in lanes]
        outputs, timing = generate(role, r, prompts, batch)
        assert len(outputs) == len(lanes)
        for lane, result in zip(lanes, outputs):
            lane['feedbacks' if role == 'feedback' else 'answers'].append(result)
        measurements.append(timing)
    for lane in lanes:
        lane['terminal_parse'] = parse_answer(lane['answers'][1]['raw_text'])
    result = dict(batch_id=batch['batch_id'], rep_seed=batch['rep_seed'], plan_sha256=sha(ROOT / 'plan.json'),
                  status='complete', lanes=lanes, measurements=measurements, inference_property_calls=0)
    audit_batch(result, batch, templates)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--index', type=int, default=int(os.environ.get('SLURM_ARRAY_TASK_ID', '0')))
    p.add_argument('--max-batches', type=int)
    args = p.parse_args()
    require_allocation()
    verify()
    assert 0 <= args.index < 17
    seed = 51004 + args.index
    directory = ROOT / 'generation' / str(seed)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        attempt = directory / 'attempts' / os.environ['SLURM_JOB_ID']
        attempt.mkdir(parents=True, exist_ok=False)
        os.environ.update(isolate(attempt))
        started = time.perf_counter()
        status = dict(seed=seed, job_id=os.environ['SLURM_JOB_ID'], status='running', completed_trajectories=0)
        save(attempt / 'started.json', status)
        try:
            backend = Backend(plan(), directory, attempt)
            templates = read(LEGACY / 'configs/prompts.json')
            seconds = 0
            for idx, batch in enumerate(batches(seed)):
                if args.max_batches is not None and idx >= args.max_batches:
                    break
                dest = directory / 'batches' / (batch['batch_id'] + '.json.gz')
                if dest.exists():
                    result = read(dest)
                    audit_batch(result, batch, templates)
                else:
                    result = execute_batch(batch, backend.generate, templates)
                    save(dest, result)
                status['completed_trajectories'] += len(result['lanes'])
                seconds += sum(m['cuda_wall_seconds'] for m in result['measurements'])
                status['generate_cuda_wall_seconds'] = seconds
                save(directory / 'progress.json', status, replace=True)
                print(json.dumps(dict(event='batch_complete', **status)), flush=True)
            assert not any(n == 'rdkit' or n.startswith('rdkit.') or n == 'admet_ai' for n in sys.modules)
            if args.max_batches is None:
                assert status['completed_trajectories'] == 3905
                status.update(status='complete', model_sequence_calls=3905*4, visible_slots=3905)
                if not (directory / 'complete.json').exists():
                    save(directory / 'complete.json', status)
            else:
                status['status'] = 'engineering_probe_complete'
        except BaseException as error:
            status.update(status='failed', exception=repr(error), traceback=traceback.format_exc())
            raise
        finally:
            status['attempt_wall_seconds'] = time.perf_counter() - started
            save(attempt / 'status.json', status)
            print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
