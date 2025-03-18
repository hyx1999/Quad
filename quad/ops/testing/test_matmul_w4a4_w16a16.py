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
def tl_fuse_w4a4_w16a16_matmul(  # INT4 x INT4 (K >> 64) + FP16 x FP16 (K <= 64)
    M,
    N,
    K,
    R,
    in_dtype,
    out_dtype,
    accum_dtype,
):
    assert in_dtype in [
        "float16",
        "int8",
    ], "Currently only float16 and int8 are supported"
    assert out_dtype in [
        "float16",
        "float32",
        "int32",
    ], "Currently only float16, float32 and int32 are supported"

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
    stage = 1

    block_M = block_row_warps * warp_row_tiles
    block_N = block_col_warps * warp_col_tiles
    block_K = chunk
    block_R = chunk

    A_shape = (M, K)  # int8 storage represents int4*2
    B_shape = (N, K)  # int8 storage represents int4*2
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
        A_scale: T.Buffer((M,), out_dtype),  # pyright: ignore
        B: T.Buffer(B_shape, in_dtype),  # pyright: ignore
        B_scale: T.Buffer((N,), out_dtype),  # pyright: ignore
        B_bias: T.Buffer((N,), out_dtype),  # pyright: ignore
        Ar: T.Buffer((M, R), out_dtype),  # pyright: ignore
        Br: T.Buffer((N, R), out_dtype),  # pyright: ignore
        C: T.Buffer((M, N), out_dtype),  # pyright: ignore
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bn, bm):

            A_shared = T.alloc_shared(A_shared_shape, in_dtype, scope=shared_scope)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype, scope=shared_scope)
            C_shared = T.alloc_shared(C_shared_shape, "float32", scope=shared_scope)
            A_local = T.alloc_local((warp_rows * local_size_a), in_dtype)
            B_local = T.alloc_local((warp_cols * local_size_b), in_dtype)
            C_local = T.alloc_local((warp_rows * warp_cols * local_size_c), accum_dtype)
            Ar_shared = T.alloc_shared((block_M, block_R), out_dtype, scope=shared_scope)
            Br_shared = T.alloc_shared((block_N, block_R), out_dtype, scope=shared_scope)
            Cr_local = T.alloc_fragment((block_M, block_N), "float32")
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
                c = C_shared[
                    i // micro_size_x,
                    j // micro_size_y,
                    i % micro_size_x,
                    j % micro_size_y,
                ]
                c_s = T.cast(c * A_s_sr[i] * B_s_sr[j] + B_b_sr[j] + Cr_local[i, j], out_dtype)
                C[bm * block_M + i, bn * block_N + j] = c_s

    return main

def assert_tl_fuse_w4a4_w16a16_matmul_correctness(M, N, K, R, in_dtype, out_dtype, accum_dtype):
    program = tl_fuse_w4a4_w16a16_matmul(M, N, K, R, in_dtype, out_dtype, accum_dtype)        
    matmul_kernel = tl.compile(program, target="cuda", execution_backend="cython")
    profiler = matmul_kernel.get_profiler()

    A = torch.randint(0, 4, (M, K), device="cuda", dtype=getattr(torch, in_dtype))
    B = torch.randint(0, 4, (N, K), device="cuda", dtype=getattr(torch, in_dtype))
    Ar = torch.randn((M, R), device="cuda", dtype=getattr(torch, out_dtype))
    Br = torch.randn((N, R), device="cuda", dtype=getattr(torch, out_dtype))
    A_scale = torch.randn((M,), device="cuda", dtype=getattr(torch, out_dtype))
    B_scale = torch.randn((N,), device="cuda", dtype=getattr(torch, out_dtype))
    B_bias = torch.randn((N,), device="cuda", dtype=getattr(torch, out_dtype))
    C = torch.zeros(M, N, device="cuda", dtype=getattr(torch, out_dtype))

    compressed_A = (A[:, ::2] & 0x0F) + ((A[:, 1::2] & 0x0F) << 4)
    compressed_B = (B[:, ::2] & 0x0F) + ((B[:, 1::2] & 0x0F) << 4)
    matmul_kernel(compressed_A, A_scale, compressed_B, B_scale, B_bias, Ar, Br, C)
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
    assert_tl_fuse_w4a4_w16a16_matmul_correctness(16, 4096, 4096, 128, "int8", "float32", "int32")


if __name__ == "__main__":
    test_assert_tl_matmul_correctness()
