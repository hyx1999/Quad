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
from collections import defaultdict

kernel_cache = defaultdict(lambda: None)

@simplify_prim_func
def tl_w4a4_matmul(  # INT4 x INT4 (K >> 64) + FP16 x FP16 (K <= 64)
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    accum_dtype,
):
    K = K // 2

    micro_size_x = micro_size_y = micro_size_k = 16

    if accum_dtype == "int32":
        micro_size_k = 32

    # This is a debug config
    block_row_warps = 2
    block_col_warps = 2
    warp_row_tiles = 32
    warp_col_tiles = 32
    chunk = 16 if in_dtype == "float16" else 32
    shared_scope = "shared.dyn"

    # Pipeline Stage
    stage = 2

    block_M = block_row_warps * warp_row_tiles
    block_N = block_col_warps * warp_col_tiles
    block_K = chunk

    A_shape = (M, K)  # int8 storage represents int4*2
    B_shape = (N, K)  # int8 storage represents int4*2
    A_s_shape = (M,)
    B_s_shape = B_b_shape = (N,)
    A_shared_shape = (block_M, block_K)
    B_shared_shape = (block_N, block_K)
    C_shared_shape = (
        block_M // micro_size_x,
        block_N // micro_size_y,
        micro_size_x,
        micro_size_y,
    )

    warp_size = 32
    threads = warp_size * (block_row_warps * block_col_warps)
    local_size_a = (micro_size_x * micro_size_k) // warp_size
    local_size_b = (micro_size_y * micro_size_k) // warp_size
    local_size_c = (micro_size_x * micro_size_y) // warp_size
    warp_rows = warp_row_tiles // micro_size_x
    warp_cols = warp_col_tiles // micro_size_y
    
    # MMA Wrapper to Auto Generate Code for MMA
    mma_emitter = INT4TensorCoreIntrinEmitter(
        a_dtype=in_dtype,
        b_dtype=in_dtype,
        accum_dtype=accum_dtype,
        a_transposed=False,
        b_transposed=True,
        block_row_warps=block_row_warps,
        block_col_warps=block_col_warps,
        warp_row_tiles=warp_row_tiles,
        warp_col_tiles=warp_col_tiles,
        chunk=chunk,
    )

    @T.prim_func
    def main(
        A: T.Buffer(A_shape, in_dtype),  # pyright: ignore
        A_scale: T.Buffer(A_s_shape, out_dtype),  # pyright: ignore
        B: T.Buffer(B_shape, in_dtype),  # pyright: ignore
        B_scale: T.Buffer(B_s_shape, out_dtype),  # pyright: ignore
        B_bias: T.Buffer(B_b_shape, out_dtype),  # pyright: ignore
        C: T.Buffer((M, N), out_dtype),  # pyright: ignore
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bn, bm):

            A_shared = T.alloc_shared(A_shared_shape, in_dtype, scope=shared_scope)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype, scope=shared_scope)
            C_shared = T.alloc_shared(C_shared_shape, "float32", scope=shared_scope)
            A_local = T.alloc_local((warp_rows * local_size_a), in_dtype)
            B_local = T.alloc_local((warp_cols * local_size_b), in_dtype)
            C_local = T.alloc_local((warp_rows * warp_cols * local_size_c), accum_dtype)
            A_s_sr = T.alloc_shared((block_M,), dtype="float32")
            B_s_sr = T.alloc_shared((block_N,), dtype="float32")
            B_b_sr = T.alloc_shared((block_N,), dtype="float32")

            T.annotate_layout({
                A_shared: make_swizzle_layout(A_shared),
                B_shared: make_swizzle_layout(B_shared),
            })

            # Improve L2 Cache
            T.use_swizzle(panel_size=10)
            
            T.copy(A_scale[bm * block_M:(bm + 1) * block_M], A_s_sr)
            T.copy(B_scale[bn * block_N:(bn + 1) * block_N], B_s_sr)
            T.copy(B_bias[bn * block_N:(bn + 1) * block_N], B_b_sr)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=stage):

                # Load A into shared memory
                for i, k in T.Parallel(block_M, block_K):
                    A_shared[i, k] = A[bm * block_M + i, ko * block_K + k]

                # Load B into shared memory
                for j, k in T.Parallel(block_N, block_K):
                    B_shared[j, k] = B[bn * block_N + j, ko * block_K + k]

                for ki in T.serial(0, (block_K // micro_size_k)):

                    # Load A into fragment
                    mma_emitter.ldmatrix_a(
                        A_local,
                        A_shared,
                        ki,
                    )

                    # Load B into fragment
                    mma_emitter.ldmatrix_b(
                        B_local,
                        B_shared,
                        ki,
                    )

                    # Perform Matrix Multiplication
                    mma_emitter.mma(A_local, B_local, C_local)

            # Perform STMatrix
            mma_emitter.stmatrix(
                C_local,
                C_shared,
            )
            
            # Store shared into global
            for i, j in T.Parallel(block_M, block_N):
                c = C_shared[
                    i // micro_size_x,
                    j // micro_size_y,
                    i % micro_size_x,
                    j % micro_size_y,
                ]
                c_s = T.cast(c * A_s_sr[i] * B_s_sr[j] + B_b_sr[j], out_dtype)
                C[bm * block_M + i, bn * block_N + j] = c_s

    return main


def get_w4a4_matmul_kernel(
    in_features: int,
    out_features: int, 
):
    in_dtype: str = "int8",
    out_dtype: str = "float16",
    accum_dtype: str = "int32"
    
    if kernel_cache[(in_features, out_features)] is None:
        print("init tl_w4a4_matmul kernel...")
        program = tl_w4a4_matmul(
            T.symbolic("num_tokens"),
            out_features,
            in_features,
            in_dtype,
            out_dtype,
            accum_dtype
        )
        
        matmul_kernel = tl.compile(program, out_idx=-1, target="cuda", execution_backend="cython")
        kernel_cache[(in_features, out_features)] = matmul_kernel
    else:
        matmul_kernel = kernel_cache[(in_features, out_features)]
    
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
