"""One standard P6 assembly; independent per-G, two-wave persistent RNG streams."""
import copy
import hashlib
import pickle
import random
import time
import zlib
from collections import Counter
from common import ROOT, CONTRACT, append, write, read, text_sha, require_allocation


class RNGArchive:
    def __init__(self, path):
        self.file = path.open('xb')
        self.refs = {}

    def save(self, state):
        data = pickle.dumps(state, protocol=4)
        digest = hashlib.sha256(data).hexdigest()
        if digest not in self.refs:
            packed = zlib.compress(data, 1)
            offset = self.file.tell()
            self.file.write(packed)
            self.file.flush()
            self.refs[digest] = {'sha256': digest, 'offset': offset, 'compressed_bytes': len(packed),
                                 'codec': 'zlib+pickle4; CPU/CUDA RNG tensors as bytes; numpy keys as list'}
        return self.refs[digest]

    def close(self):
        self.file.close()


def rng_capture():
    import torch
    import numpy as np
    ns = np.random.get_state()
    return {'python': random.getstate(), 'numpy': (ns[0], ns[1].tolist(), ns[2], ns[3], ns[4]),
            'torch_cpu': bytes(torch.get_rng_state().tolist()),
            'torch_cuda': [bytes(s.tolist()) for s in torch.cuda.get_rng_state_all()]}


def rng_restore(state):
    import torch
    import numpy as np
    random.setstate(state['python'])
    ns = state['numpy']
    np.random.set_state((ns[0], np.array(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
    torch.set_rng_state(torch.tensor(list(state['torch_cpu']), dtype=torch.uint8))
    torch.cuda.set_rng_state_all([torch.tensor(list(s), dtype=torch.uint8) for s in state['torch_cuda']])


class Backend:
    def __init__(self, directory, unit):
        self.directory, self.unit = directory, unit
        self.model = None
        self.archive = RNGArchive(directory / 'rng_states.bin')
        self.rng, self.G = {}, None
        self.seen_G = []
        self.cost = {'model_load_started': 0, 'model_load_completed': 0, 'model_load_seconds': None,
                     'calls_started': 0, 'calls_completed': 0, 'by_G': {}}

    def checkpoint(self):
        write(self.directory / 'generation_cost.json', self.cost, replace=True)

    def start(self):
        require_allocation()
        import torch, transformers, peft
        from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
        from peft import PeftModel
        from dataclasses import asdict
        assert self.model is None
        if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
            raise RuntimeError('Expected one BF16 Blackwell GPU')
        torch.cuda.set_device(0)
        torch.set_num_threads(8)
        gpu = torch.cuda.get_device_properties(0)
        if gpu.name != 'NVIDIA RTX PRO 6000 Blackwell Server Edition' or gpu.total_memory < 40 * 1024**3:
            raise RuntimeError('GPU differs from approved Blackwell')
        if (torch.__version__, transformers.__version__, peft.__version__) != ('2.7.0+cu128', '4.46.3', '0.14.0'):
            raise RuntimeError('Generation environment changed')
        spec = read(ROOT / 'sources/runtime.json')
        base_path, adapter_path = spec['base_path'], spec['adapter_path']
        tokenizer_started = time.perf_counter()
        self.tok = AutoTokenizer.from_pretrained(base_path, use_fast=False, local_files_only=True, trust_remote_code=False)
        self.cost['tokenizer_load_seconds'] = time.perf_counter() - tokenizer_started
        assert not self.tok.is_fast and self.tok.padding_side == 'right'
        assert (self.tok.bos_token_id, self.tok.eos_token_id) == (1, 2)
        assert self.tok.add_bos_token is True and self.tok.add_eos_token is False
        if self.tok.pad_token_id is None:
            self.tok.pad_token_id = self.tok.eos_token_id
        assert self.tok.pad_token_id == 2
        self.base_generation = GenerationConfig.from_pretrained(base_path, local_files_only=True)
        d = CONTRACT['decoding_contract']
        keys = ['do_sample', 'temperature', 'top_p', 'top_k', 'num_beams', 'num_beam_groups', 'num_return_sequences',
                'repetition_penalty', 'no_repeat_ngram_size', 'length_penalty', 'min_new_tokens', 'min_length']
        opts = {k: d[k] for k in keys}
        opts.update(bos_token_id=1, eos_token_id=2, pad_token_id=2, use_cache=True,
                    return_dict_in_generate=False, forced_eos_token_id=None, forced_bos_token_id=None,
                    stop_strings=None)
        self.configs = {}
        for wave, cap in [(0, 200), (1, 220)]:
            cfg = copy.deepcopy(self.base_generation)
            unused = cfg.update(max_new_tokens=cap, **opts)
            assert not unused
            cfg.validate()
            self.configs[wave] = cfg
        identity = {'gpu': {'name': gpu.name, 'total_memory_bytes': gpu.total_memory, 'capability': [gpu.major, gpu.minor]},
                    'versions': {'torch': torch.__version__, 'transformers': transformers.__version__, 'peft': peft.__version__},
                    'tokenizer': {'class': type(self.tok).__name__, 'is_fast': self.tok.is_fast, 'padding_side': self.tok.padding_side,
                                  'bos': 1, 'eos': 2, 'pad': 2, 'add_bos': self.tok.add_bos_token, 'add_eos': self.tok.add_eos_token,
                                  'chat_template_applied': False, 'size': len(self.tok)},
                    'base_generation_config': self.base_generation.to_dict(), 'runtime_spec': spec,
                    'effective_configs': {str(k): v.to_dict() for k, v in self.configs.items()},
                    'other_generate_kwargs': {'synced_gpus': False}, 'rng_seeded_after_model_assembly': True}
        write(self.directory / 'initialization.json', identity)
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.synchronize(0)
        t = time.perf_counter()
        self.cost['model_load_started'] = 1
        self.checkpoint()
        try:
            base = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16, device_map={'': 'cuda:0'},
                       local_files_only=True, trust_remote_code=False, use_safetensors=True)
            self.model = PeftModel.from_pretrained(base, adapter_path, adapter_name='default', is_trainable=False,
                       autocast_adapter_dtype=True, low_cpu_mem_usage=False, ephemeral_gpu_offload=False, local_files_only=True)
            self.model.eval()
            torch.cuda.synchronize(0)
            self.cost['model_load_completed'] = 1
        finally:
            self.cost['model_load_seconds'] = time.perf_counter() - t
            self.checkpoint()
        status = asdict(self.model.get_model_status())
        assert status['enabled'] is True and status['active_adapters'] == ['default'] and status['merged_adapters'] == []
        placements = Counter((str(p.device), str(p.dtype), 'lora' if 'lora_' in n else 'base') for n, p in self.model.named_parameters())
        assert all(device == 'cuda:0' for device, _, _ in placements)
        assert all(dtype == ('torch.float32' if kind == 'lora' else 'torch.bfloat16') for _, dtype, kind in placements)
        identity.update(adapter_status=status, parameter_placements=[{'device': k[0], 'dtype': k[1], 'kind': k[2], 'parameter_tensors': v} for k, v in placements.items()],
                        attention_implementation=self.model.config._attn_implementation)
        write(self.directory / 'model_identity.json', identity)

    def begin_G(self, G):
        import numpy as np
        import torch
        assert G == [12, 6, 24][len(self.seen_G)] and self.model is not None
        self.seen_G.append(G)
        self.G = G
        self.rng = {}
        self.cost['by_G'][str(G)] = {'calls_started': 0, 'calls_completed': 0, 'generate_seconds': 0.0,
                                       'prompt_tokens': 0, 'output_tokens_through_eos': 0, 'cap_hits': 0}
        for wave in (0, 1):
            seed = self.unit['wave{}_seed'.format(wave)]
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            state = rng_capture()
            self.rng[wave] = state
            append(self.directory / 'rng_initializations.jsonl', {'G': G, 'wave': wave, 'seed': seed, 'state': self.archive.save(state)})

    def query(self, prompt, num_responses, spec):
        assert self.G == spec['G'] and num_responses == (self.G if spec['wave'] == 0 else 1)
        wave = spec['wave']
        rng_restore(self.rng[wave])
        responses = []
        for i in range(num_responses):
            context = {k: v for k, v in spec.items() if k != 'prompt'}
            if wave == 0:
                context['local_node_id'] = 'g{:02d}'.format(i)
            responses.append(self.generate(prompt, context, self.configs[wave]))
        self.rng[wave] = rng_capture()
        return responses

    def generate(self, prompt, context, cfg):
        import torch
        ids = self.tok(prompt, add_special_tokens=True, truncation=False, return_attention_mask=True)
        input_ids = ids['input_ids']
        call_index = self.cost['calls_started']
        call = {'call_index': call_index, 'context': context, 'prompt': prompt, 'prompt_sha256': text_sha(prompt),
                'input_ids': input_ids, 'attention_mask': ids['attention_mask'], 'prompt_tokens': len(input_ids),
                'base_generation_config': self.base_generation.to_dict(), 'generation_config_passed': cfg.to_dict(),
                'other_generate_kwargs': {'synced_gpus': False}, 'started': False, 'completed': False,
                'exception': None, 'started_unix': None, 'cuda_wall_seconds': None}
        if not input_ids or len(input_ids) > 4096 or len(input_ids) + cfg.max_new_tokens > 32768 or any(v != 1 for v in ids['attention_mask']):
            call['exception'] = 'input_contract_overflow_or_invalid; no model call or truncation'
            append(self.directory / 'calls.jsonl', dict(call, event='precall_failure'))
            raise ValueError(call['exception'])
        before = rng_capture()
        call['rng_before'] = self.archive.save(before)
        tokens = {k: torch.tensor([v], dtype=torch.long, device='cuda:0') for k, v in ids.items()}
        torch.cuda.synchronize(0)
        t = time.perf_counter()
        call.update(started=True, started_unix=time.time())
        self.cost['calls_started'] += 1
        g = self.cost['by_G'][str(self.G)]
        g['calls_started'] += 1
        append(self.directory / 'calls.jsonl', dict(call, event='started'))
        t = time.perf_counter()
        try:
            with torch.inference_mode():
                sequence = self.model.generate(**tokens, generation_config=cfg, synced_gpus=False)
            torch.cuda.synchronize(0)
            call['cuda_wall_seconds'] = time.perf_counter() - t
            call['model_generate_returned'] = True
            rows = sequence.detach().cpu().tolist()
            assert len(rows) == 1 and rows[0][:len(input_ids)] == input_ids
            continuation = rows[0][len(input_ids):]
            assert 1 <= len(continuation) <= cfg.max_new_tokens
            counted = continuation.index(2) + 1 if 2 in continuation else len(continuation)
            raw = self.tok.decode(continuation, skip_special_tokens=True)
            call.update(completed=True, continuation_ids=continuation, tokens_through_first_eos=counted,
                        post_eos_tokens=len(continuation) - counted, reached_cap_without_eos=counted == cfg.max_new_tokens and 2 not in continuation,
                        decoded_raw_with_special_tokens=self.tok.decode(continuation, skip_special_tokens=False, clean_up_tokenization_spaces=False),
                        decoded_continuation_untrimmed=raw)
            self.cost['calls_completed'] += 1
            g['calls_completed'] += 1
            g['prompt_tokens'] += len(input_ids)
            g['output_tokens_through_eos'] += counted
            g['cap_hits'] += int(call['reached_cap_without_eos'])
            return raw
        except BaseException as err:
            call['exception'] = repr(err)
            raise
        finally:
            if call['cuda_wall_seconds'] is None:
                try:
                    torch.cuda.synchronize(0)
                except BaseException as err:
                    call['synchronize_error'] = repr(err)
                call['cuda_wall_seconds'] = time.perf_counter() - t
            try:
                call['rng_after'] = self.archive.save(rng_capture())
            except BaseException as err:
                call['rng_after_error'] = repr(err)
            g['generate_seconds'] += call['cuda_wall_seconds']
            append(self.directory / 'calls.jsonl', dict(call, event='result'))
            self.checkpoint()

    def close(self):
        if self.model is not None:
            import torch
            self.cost['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(0)
            self.cost['peak_reserved_bytes'] = torch.cuda.max_memory_reserved(0)
        self.checkpoint()
        self.archive.close()
