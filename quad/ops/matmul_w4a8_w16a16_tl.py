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
    make_mma_swizzle_layout as make_swizzle_layout,
)
from tilelang.transform import simplify_prim_func
from collections import defaultdict

kernel_cache = defaultdict(lambda: None)

@simplify_prim_func
def tl_fuse_w4a8_w16a16_matmul(  # INT4 x INT4 (K >> 64) + FP16 x FP16 (K <= 64)
    M,
    N,
    K,
    R,
    A_dtype="int8",
    B_dtype="int8",
    out_dtype="float16",
    accum_dtype="int32",
):
    B_deq_dtype = A_dtype
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
    Ar_shape = (M, R)
    Br_shape = (N, R)
    A_s_shape = (M,)
    B_s_shape = B_b_shape = (N,)
    A_shared_shape = (block_M, block_K)
    B_shared_shape = (block_N, block_K // 2)
    B_deq_shape = (block_N, block_K)
    Ar_shared_shape = (block_M, block_R)
    Br_shared_shape = (block_N, block_R)
    C_shared_shape = (block_M, block_N)

    @T.prim_func
    def main(
        A: T.Buffer(A_shape, in_dtype),  # pyright: ignore
        A_scale: T.Buffer(A_s_shape, out_dtype),  # pyright: ignore
        B: T.Buffer(B_shape, "uint8"),  # pyright: ignore
        B_scale: T.Buffer(B_s_shape, out_dtype),  # pyright: ignore
        B_bias: T.Buffer(B_b_shape, out_dtype),  # pyright: ignore
        Ar: T.Buffer(Ar_shape, out_dtype),  # pyright: ignore
        Br: T.Buffer(Br_shape, out_dtype),  # pyright: ignore
        C: T.Buffer((M, N), out_dtype),  # pyright: ignore
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bn, bm):

            A_shared = T.alloc_shared(A_shared_shape, A_dtype, scope=shared_scope)
            B_shared = T.alloc_shared(B_shared_shape, B_dtype, scope=shared_scope)
            B_deq_shared = T.alloc_shared(B_deq_shape, B_deq_dtype, scope=shared_scope)
            C_local = T.alloc_fragment(C_shared_shape, accum_dtype)
            A_s_sr = T.alloc_shared((block_M,), dtype="float32")
            B_s_sr = T.alloc_shared((block_N,), dtype="float32")
            B_b_sr = T.alloc_shared((block_N,), dtype="float32")
            Ar_shared = T.alloc_shared(Ar_shared_shape, out_dtype, scope=shared_scope)
            Br_shared = T.alloc_shared(Br_shared_shape, out_dtype, scope=shared_scope)
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
                T.copy(B[bn * block_N, ko * (block_K // 2)], B_shared)
                
                for i, j in T.Parallel(block_N, block_K):
                    if j % 2 == 0:
                        x_i4 = ((B_shared[i, j // 2] & T.cast(0x0F, "uint8")) >> 0)
                        x_i8 = (x_i4 ^ 0x8) - 0x8
                        B_deq_shared[i, j] = x_i8
                    else:
                        x_i4 = ((B_shared[i, j // 2] & T.cast(0xF0, "uint8")) >> 4)
                        x_i8 = (x_i4 ^ 0x8) - 0x8
                        B_deq_shared[i, j] = x_i8
                
                # Perform local GEMM on the shared-memory tiles
                T.gemm(A_shared, B_deq_shared, C_local, transpose_B=True)
            
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


def get_fuse_w4a8_w16a16_matmul_kernel(
    in_features: int,
    out_features: int, 
    rank: int,
):
    A_dtype: str = "int8"
    B_dtype: str = "int8"
    out_dtype: str = "float16"
    accum_dtype: str = "int32"
    
    if kernel_cache[(in_features, out_features, rank)] is None:
        print("init tl_fuse_w4a8_w16a16_matmul kernel...")
        program = tl_fuse_w4a8_w16a16_matmul(
            T.symbolic("num_tokens"),
            out_features,
            in_features,
            rank,
            A_dtype,
            B_dtype,
            out_dtype,
            accum_dtype
        )
        
        matmul_kernel = tl.compile(program, out_idx=-1, target="cuda", execution_backend="cython")
        kernel_cache[(in_features, out_features, rank)] = matmul_kernel
    else:
        matmul_kernel = kernel_cache[(in_features, out_features, rank)]

    return matmul_kernel
    # @torch.no_grad()
    # def matmul_fn(x_quant, x_scale, x_r, weight_quant, weight_scale, weight_r, bias):
    #     return matmul_kernel(
    #         x_quant,
    #         x_scale,
    #         weight_quant,
    #         weight_scale,
    #         bias,
    #         x_r,
    #         weight_r
    #     )
