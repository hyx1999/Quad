import argparse
import gc
import pprint
import numpy as np
import torch
import time
import torch
import transformers
from quad_svd.models.qwen2.quad_qwen2_tl import QuadQwen2Config, QuadQwen2ForCausalLM
from tqdm import tqdm

model_configs = [
    ("misc/checkpoints/Qwen2.5-7B-quadsvd-svd64-sa25", "/data/models/Qwen2.5-7B"),
]

benchmark_dtypes = ["int4", torch.float16]
num_warmup_steps = 0
num_bench_steps = 1

def repeated_run(num_repeats=10, warmup_repeats=2):
    def func(module):
        def _f(*args, **kwargs):
            for i in tqdm(range(warmup_repeats), desc="warmup..."):
                module(*args, **kwargs)
            times = []
            for i in tqdm(range(num_repeats), desc="repeats..."):
                times.append(module(*args, **kwargs))
            return tuple(zip(*times))
        return _f
    return func

def _cleanup():
    gc.collect()
    torch.cuda.empty_cache()

@repeated_run()
def module_benchmark(module):
    # warmup
    for i in range(num_warmup_steps):
        out = module()
    torch.cuda.synchronize()
    
    _cleanup()
    torch.cuda.reset_max_memory_allocated()
    start_time = time.perf_counter()
    
    
    for i in range(num_bench_steps):
        out = module()
    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated()

    end_time = time.perf_counter()

    return (end_time - start_time) * 1000 / num_bench_steps, peak_memory


def get_model_quantized(config_name):
    config: QuadQwen2Config = QuadQwen2Config.from_pretrained(config_name)
    config._attn_implementation = "flash_attention_2"
    with transformers.modeling_utils.no_init_weights():
        model = QuadQwen2ForCausalLM(config=config)
        model.half()
    for name, buffer in model.named_buffers():
        if buffer.dtype == torch.uint8:
            buffer.data = buffer.data.to(torch.int8)
    model.to("cuda")
    print("device:", model.device)
    return model


def get_model_hf(config_name):
    return transformers.Qwen2ForCausalLM.from_pretrained(
        config_name, 
        torch_dtype=torch.float16, 
        attn_implementation="flash_attention_2"
    ).to("cuda")


def run_prefill(model, bsz, prefill_length):
    device = model.device
    test_input = torch.randint(100, 200, (bsz, prefill_length), dtype=torch.int32, device=device)
    return module_benchmark(lambda: model(test_input))


def run_decode(model, bsz, prefill_length, decode_steps):
    device = model.device
    test_input = torch.randint(100, 200, (bsz, prefill_length), dtype=torch.int32, device=device)
    model._expected_max_length = prefill_length + decode_steps
    out = model(test_input)
    past_key_values = out.past_key_values
    del out
    _cleanup()
    next_input = torch.tensor([[100] for _ in range (bsz)], dtype=torch.int32, device=device)
    def _decode_for_multiple_steps():
        # past_key_values.length = prefill_length
        for _ in range(decode_steps):
            model(next_input, past_key_values=past_key_values)
    return module_benchmark(_decode_for_multiple_steps)
    

def run_e2e(model, bsz, prefill_length, decode_steps):
    device = model.device
    test_input = torch.randint(100, 200, (bsz, prefill_length), dtype=torch.int32, device=device)
    next_input = torch.tensor([[100] for _ in range (bsz)], dtype=torch.int32, device=device)
    def _prefill_and_decode_for_multiple_steps():
        model._expected_max_length = prefill_length + decode_steps
        out = model(test_input)
        for _ in range(decode_steps):
            model(next_input, past_key_values=out.past_key_values)
    return module_benchmark(_prefill_and_decode_for_multiple_steps)


def _wait_for_input():
    print("Press enter")
    input()

@torch.no_grad
def run_all_for_model(model, bsz, prefill, decode):
    model.eval()
    model = model.cuda()
    time_prefill, _ = run_prefill(model, bsz, prefill)
    _cleanup()
    if decode is not None:
        time_decode, memory_decode = run_decode(model, bsz, prefill, decode)
        _cleanup()
        time_e2e, _ = run_e2e(model, bsz, prefill, decode)
        _cleanup()
    else:
        time_decode = time_e2e = memory_decode = None
    return time_prefill, time_decode, time_e2e, memory_decode

def benchmark(args):
    
    for config_names in model_configs:
        quad_config_name = config_names[0]
        hf_config_name = config_names[1]

        model = get_model_quantized(quad_config_name)
        time_prefill_i4, time_decode_i4, time_e2e_i4, mem_i4 = run_all_for_model(
            model, args.batch_size, args.prefill_seq_len, args.decode_steps)
        del model
        _cleanup()

        model = get_model_hf(hf_config_name)
        time_prefill_f16, time_decode_f16, time_e2e_f16, mem_f16 = run_all_for_model(
            model, args.batch_size, args.prefill_seq_len, args.decode_steps)
        del model
        _cleanup()


        print(f"Prefill Int4 time: {np.mean(time_prefill_i4):.3f} +- {1.96 * np.std(time_prefill_i4):.3f}ms")
        print(f"Prefill FP16 time: {np.mean(time_prefill_f16):.3f} +- {1.96 * np.std(time_prefill_f16):.3f}ms")
        print(f"Speedup: {np.mean(time_prefill_f16) / np.mean(time_prefill_i4):.3f}x")
        print(f'Prefill & {hf_config_name} & {args.batch_size} & {args.prefill_seq_len} & {np.mean(time_prefill_f16):.3f} & {np.mean(time_prefill_i4):.3f}\\\\')

        if args.decode_steps is not None:
            print(f"Decode Int4 time: {np.mean(time_decode_i4):.3f} +- {1.96 * np.std(time_decode_i4):.3f}ms")
            print(f"Decode FP16 time: {np.mean(time_decode_f16):.3f} +- {1.96 * np.std(time_decode_f16):.3f}ms")
            print(f"Speedup: {np.mean(time_decode_f16) / np.mean(time_decode_i4):.3f}x")
            print(f'Decode & {hf_config_name} & {args.batch_size} & {args.prefill_seq_len} & {args.decode_steps} & {np.mean(time_decode_f16):.3f} & {np.mean(time_decode_i4):.3f}\\\\')

            print(f"E2E Int4 time: {np.mean(time_e2e_i4):.3f} +- {1.96 * np.std(time_e2e_i4):.3f}ms")
            print(f"E2E FP16 time: {np.mean(time_e2e_f16):.3f} +- {1.96 * np.std(time_e2e_f16):.3f}ms")
            print(f"Speedup: {np.mean(time_e2e_f16) / np.mean(time_e2e_i4):.3f}x")
            print(f'E2E & {hf_config_name} & {args.batch_size} & {args.prefill_seq_len} & {args.decode_steps} & {np.mean(time_e2e_f16):.3f} & {np.mean(time_e2e_i4):.3f}\\\\')
            
            # table-style output

            print(f"Int4 memory: {np.mean(mem_i4) / (1024 * 1024 * 1024):.3f}GB +- {1.96 * np.std(mem_i4):.3f}")
            print(f"FP16 memory: {np.mean(mem_f16) / (1024 * 1024 * 1024):.3f}GB +- {1.96 * np.std(mem_f16):.3f}")
            print(f"Memory saving: {np.mean(mem_f16) / np.mean(mem_i4):.3f}x")
            print(f'Memory saving & {hf_config_name} & {args.batch_size} & {args.prefill_seq_len} & {args.decode_steps} & {np.mean(mem_i4) / (1024 * 1024 * 1024):.3f}GB & {np.mean(mem_f16) / (1024 * 1024 * 1024):.3f}GB\\\\')
        
        print('--------------')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--batch_size', type=int,
        help='Batch size',
        default=1,
    )
    parser.add_argument(
        '--prefill_seq_len', type=int,
        help='Size of the input sequence',
        default=2048,
    )
    parser.add_argument(
        '--decode_steps', type=int,
        help='Decode steps',
        default=None,
    )
    
    args = parser.parse_args()
    pprint.pprint(vars(args))
    benchmark(args)
