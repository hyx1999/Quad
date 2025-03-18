import torch
import time
import argparse
import numpy as np
import pprint
from quad.ops.matmul_w4a4_w16a16_tl import get_fuse_w4a4_w16a16_matmul_kernel
from quad.ops.matmul_w4a8_tl import get_w4a8_matmul_kernel
import quad_cuda

model_sizes = [
    (4096, 4096), #llama-7b
    (5120, 5120), #llama-13b
    (8192, 8192)  #llama-70b   
]

mlp_sizes = [
    (4096, 11008), #llama-7b
    (5120, 13824), #llama-13b
    (8192, 28672)  #llama-70b
]
benchmark_dtypes = [torch.float16]
num_warmup_steps = 5
num_bench_steps = 100


def module_benchmark(module, x):
    x = x.cuda()
    
    # warmup
    for i in range(num_warmup_steps):
        out = module(x)
    torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    for i in range(num_bench_steps):
        out = module(x)
    torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    
    
    return (end_time - start_time) * 1000 / num_bench_steps

def module_benchmark_w4a4(kernel, x_quant, x_scale, w_quant, w_scale, w_bias, x_r, w_r):    
    # warmup
    for i in range(num_warmup_steps):
        out = kernel(x_quant, x_scale, w_quant, w_scale, w_bias, x_r, w_r)
    torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    for i in range(num_bench_steps):
        out = kernel(x_quant, x_scale, w_quant, w_scale, w_bias, x_r, w_r)
    torch.cuda.synchronize()
    
    end_time = time.perf_counter()
        
    return (end_time - start_time) * 1000 / num_bench_steps

def module_benchmark_w4a8(kernel, x_quant, x_scale, w_quant, w_scale, w_bias, x_r, w_r):    
    # warmup
    for i in range(num_warmup_steps):
        out = quad_cuda.s8s4_linear_cutlass(x_quant, x_scale, w_quant, w_scale, w_bias)
        # out = kernel(x_quant, x_scale, w_quant, w_scale, w_bias)
    torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    for i in range(num_bench_steps):
        out = quad_cuda.s8s4_linear_cutlass(x_quant, x_scale, w_quant, w_scale, w_bias)
        # out = kernel(x_quant, x_scale, w_quant, w_scale, w_bias)
    torch.cuda.synchronize()
    
    end_time = time.perf_counter()
        
    return (end_time - start_time) * 1000 / num_bench_steps


def linear4bit_benchmark(args):

    pod_rank = 64
    bsz = args.bsz
    seq_len = args.seq_len
    
    if args.layer_type == 'v_proj':
        layer_size = model_sizes
    else:
        layer_size = mlp_sizes
        
    
    for (feature_dim_in, feature_dim_out) in layer_size:
        for dtype in benchmark_dtypes:
            
            x = torch.rand((bsz * seq_len,
                            feature_dim_in)).cuda().to(dtype)
            
            baseline_mod = torch.nn.Linear(feature_dim_in,
                                           feature_dim_out,
                                           bias=False).cuda().to(dtype)
            
            baseline_mod.weight.data = torch.randint_like(baseline_mod.weight.data,
                                                          low=-8, high=7).to(dtype)
            
            kernel = get_fuse_w4a4_w16a16_matmul_kernel(
                feature_dim_in,
                feature_dim_out,
                # pod_rank
            )
            
            x_int = torch.randint_like(x, low=0, high=4).to(torch.int8).cuda()
            w_int = torch.randint_like(baseline_mod.weight.data, low=0, high=4).to(torch.int8).cuda()
            x_quant = x_int[..., 0::2] | (x_int[..., 1::2] << 4)
            x_scale = torch.ones((bsz * seq_len,), dtype=torch.float16, device='cuda')
            w_quant = w_int[..., 0::2] | (w_int[..., 1::2] << 4)
            w_scale = torch.ones((feature_dim_out,), dtype=torch.float16, device='cuda')
            w_bias  = torch.zeros((feature_dim_out,), dtype=torch.float16, device='cuda')
            w_r = torch.randn((feature_dim_out, pod_rank), dtype=torch.float16, device='cuda')
            x_r = torch.randn((bsz * seq_len, pod_rank), dtype=torch.float16, device='cuda')

            print(f"{dtype}. Sizes: {baseline_mod.weight.shape}")
            times_w4a8 = []
            for i in range(10):
                times_w4a8.append(
                    module_benchmark_w4a8(None, x_quant, x_scale, w_quant, w_scale, w_bias, x_r, w_r)
                )
            print(f"w4a8 time: {np.mean(times_w4a8):.3f} +- {1.96 * np.std(times_w4a8):.3f}ms")

            print(f"{dtype}. Sizes: {baseline_mod.weight.shape}")
            times_w4a4_w16a16 = []
            for i in range(10):
                times_w4a4_w16a16.append(
                    module_benchmark_w4a4(kernel, x_quant, x_scale, w_quant, w_scale, w_bias, x_r, w_r)
                )
            print(f"w4a4+w16a16 time: {np.mean(times_w4a4_w16a16):.3f} +- {1.96 * np.std(times_w4a4_w16a16):.3f}ms")
            
            times_baseline = []
            for i in range(10):
                times_baseline.append(module_benchmark(baseline_mod, x))
            print(f"FP16 time: {np.mean(times_baseline):.3f} +- {1.96 * np.std(times_baseline):.3f}ms")
            
            print(f"Speedup: {np.mean(times_baseline) / np.mean(times_w4a8):.3f}x")
            print(f"Speedup: {np.mean(times_baseline) / np.mean(times_w4a4_w16a16):.3f}x")
            
            # table-style output
            # print(f'{feature_dim_in}x{feature_dim_out} & {args.bsz} & {np.mean(times_baseline):.3f} & {np.mean(times_w4a8):.3f} \\\\')
            # print('--------------')
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--bsz', type=int,
        help='Batch size',
        default=1,
    )
    parser.add_argument(
        '--seq_len', type=int,
        help='Size of the input sequence',
        default=2048,
    )
    parser.add_argument(
        '--layer_type', type=str,
        help='Type of the layer in the model (v_proj [default], down_proj)',
        default='v_proj',
        choices=['v_proj', 'down_proj']
    )
    
    args = parser.parse_args()
    pprint.pprint(vars(args))
    linear4bit_benchmark(args)
