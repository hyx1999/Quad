# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.backends
import tilelang
from tilelang import tvm as tvm
import tilelang.testing
import tilelang as tl
import tilelang.language as T
from tilelang.intrinsics import (
    make_mma_swizzle_layout as make_swizzle_layout,)

from tilelang.intrinsics.mma_macro_generator import (
    INT4TensorCoreIntrinEmitter,
    INT4TensorCoreIntrinEmitterWithLadderTransform,
)
from tilelang.transform import simplify_prim_func

tilelang.testing.set_random_seed(0)

@simplify_prim_func
def tl_fuse_w4a8_w16a16_matmul(  # INT4 x INT4 (K >> 64) + FP16 x FP16 (K <= 64)
    M,
    N,
    K,
    R,
    in_dtype="int8",
    out_dtype="float16",
    accum_dtype="int32",
):
    chunk = 32
    shared_scope = "shared.dyn"

    # Pipeline Stage
    stage = 2

    block_M = 64
    block_N = 64
    block_K = chunk
    block_R = chunk

    A_shape = (M, K)
    B_shape = (N, K // 2)
    A_shared_shape = (block_M, block_K)
    B_shared_shape = (block_N, block_K // 2)
    B_deq_shape = (block_N, block_K)
    C_shared_shape = (block_M, block_N)

    @T.prim_func
    def main(
        A: T.Buffer(A_shape, in_dtype),  # pyright: ignore
        A_scale: T.Buffer((M,), out_dtype),  # pyright: ignore
        B_pack: T.Buffer(B_shape, "uint8"),  # pyright: ignore
        B_scale: T.Buffer((N,), out_dtype),  # pyright: ignore
        B_bias: T.Buffer((N,), out_dtype),  # pyright: ignore
        Ar: T.Buffer((M, R), out_dtype),  # pyright: ignore
        Br: T.Buffer((N, R), out_dtype),  # pyright: ignore
        C: T.Buffer((M, N), out_dtype),  # pyright: ignore
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bn, bm):

            A_shared = T.alloc_shared(A_shared_shape, in_dtype, scope=shared_scope)
            B_shared = T.alloc_shared(B_shared_shape, "uint8", scope=shared_scope)
            B_local = T.alloc_fragment(B_shared_shape, "uint8")
            B_deq_local = T.alloc_shared(B_deq_shape, in_dtype)
            C_local = T.alloc_fragment(C_shared_shape, accum_dtype)
            A_s_sr = T.alloc_shared((block_M,), dtype="float32")
            B_s_sr = T.alloc_shared((block_N,), dtype="float32")
            B_b_sr = T.alloc_shared((block_N,), dtype="float32")
            Ar_shared = T.alloc_shared((block_M, block_R), out_dtype, scope=shared_scope)
            Br_shared = T.alloc_shared((block_N, block_R), out_dtype, scope=shared_scope)
            Cr_local = T.alloc_fragment((block_M, block_N), "float32")

            # Improve L2 Cache
            T.use_swizzle(panel_size=10)
            
            T.copy(A_scale[bm * block_M:(bm + 1) * block_M], A_s_sr)
            T.copy(B_scale[bn * block_N:(bn + 1) * block_N], B_s_sr)
            T.copy(B_bias[bn * block_N:(bn + 1) * block_N], B_b_sr)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=stage):

                # Copy tile of A
                T.copy(A[bm * block_M, ko * block_K], A_shared)
                # Copy tile of B
                T.copy(B_pack[bn * block_N, ko * (block_K // 2)], B_shared)
                
                T.copy(B_shared, B_local)
                for i, j in T.Parallel(block_N, block_K):
                    if j % 2 == 0:
                        x_i4 = ((B_local[i, j // 2] & T.cast(0x0F, "uint8")) >> 0)
                        x_i8 = (x_i4 ^ 0x8) - 0x8
                        B_deq_local[i, j] = x_i8
                    else:
                        x_i4 = ((B_local[i, j // 2] & T.cast(0xF0, "uint8")) >> 4)
                        x_i8 = (x_i4 ^ 0x8) - 0x8
                        B_deq_local[i, j] = x_i8
                
                # Perform local GEMM on the shared-memory tiles
                T.gemm(A_shared, B_deq_local, C_local, transpose_B=True)
            
            # Clear the local accumulation buffer
            T.clear(Cr_local)
            
            # Pipelined iteration over R dimension
            for idx in T.Pipelined(T.ceildiv(R, block_R), num_stages=stage):
                # Copy tile of A
                T.copy(Ar[bm * block_M, idx * block_R], Ar_shared)
                T.copy(Br[bn * block_N, idx * block_R], Br_shared)
                
                # Perform local GEMM on the shared-memory tiles
                T.gemm(Ar_shared, Br_shared, Cr_local, transpose_B=True)
            
            # Store shared into global
            for i, j in T.Parallel(block_M, block_N):
                c_s = T.cast(C_local[i, j] * A_s_sr[i] * B_s_sr[j] + B_b_sr[j] + Cr_local[i, j], out_dtype)
                C[bm * block_M + i, bn * block_N + j] = c_s

    return main

def assert_tl_w4a8_w16a16_matmul_correctness(M, N, K, R, in_dtype, out_dtype, accum_dtype):
    program = tl_fuse_w4a8_w16a16_matmul(M, N, K, R, in_dtype, out_dtype, accum_dtype)        
    matmul_kernel = tl.compile(program, target="cuda", execution_backend="cython")
    profiler = matmul_kernel.get_profiler()

    A = torch.randint(0, 4, (M, K), device="cuda", dtype=getattr(torch, in_dtype))
    B = torch.randint(0, 4, (N, K), device="cuda", dtype=getattr(torch, in_dtype))
    Ar = torch.randn((M, R), device="cuda", dtype=getattr(torch, out_dtype)).zero_()
    Br = torch.randn((N, R), device="cuda", dtype=getattr(torch, out_dtype)).zero_()
    A_scale = torch.randn((M,), device="cuda", dtype=getattr(torch, out_dtype)).zero_().add_(1.0)
    B_scale = torch.randn((N,), device="cuda", dtype=getattr(torch, out_dtype)).zero_().add_(1.0)
    B_bias = torch.randn((N,), device="cuda", dtype=getattr(torch, out_dtype)).zero_()
    C = torch.zeros(M, N, device="cuda", dtype=getattr(torch, out_dtype))

    compressed_B = ((B[:, ::2] & 0x0F) + ((B[:, 1::2] & 0x0F) << 4)).to(torch.uint8)
    matmul_kernel(A, A_scale, compressed_B, B_scale, B_bias, Ar, Br, C)
    print(C)

    # Get Reference Result
    def ref_program():
        ref_c = torch.matmul(A.to(torch.float32), B.T.to(torch.float32)).to(getattr(torch, accum_dtype))
        ref_c = ref_c.to(torch.float32) * A_scale[:, None].to(torch.float32) * B_scale[None, :].to(torch.float32)
        ref_c = ref_c + B_bias
        ref_c = ref_c + (Ar @ Br.T).to(torch.float32)
        ref_c = ref_c.to(getattr(torch, out_dtype))
        return ref_c

    ref_c = ref_program()
    print(ref_c)
    torch.testing.assert_close(C, ref_c, rtol=1e-2, atol=1e-2)
    
    latency = profiler.do_bench(profiler.mod, warmup=500)
    print("Tile-lang: {:.2f} ms".format(latency))

def test_assert_tl_matmul_correctness():
    # assert_tl_matmul_correctness(16, 4096, 4096, "int8", "float32", "int32")
    assert_tl_w4a8_w16a16_matmul_correctness(16, 256, 256, 64, "int8", "float32", "int32")


if __name__ == "__main__":
    test_assert_tl_matmul_correctness()
