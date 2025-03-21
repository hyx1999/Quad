# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.backends
import tilelang
from tilelang import tvm as tvm
import tilelang.testing
import tilelang as TL
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
def tl_w4a4_matmul(
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
    block_row_warps = 1
    block_col_warps = 1
    warp_row_tiles = 32
    warp_col_tiles = 32
    chunk = 32 if in_dtype == "float16" else 64
    shared_scope = "shared.dyn"

    # Pipeline Stage
    stage = 2

    block_M = block_row_warps * warp_row_tiles
    block_N = block_col_warps * warp_col_tiles
    block_K = chunk

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
            A: T.Buffer(A_shape, in_dtype),
            B: T.Buffer(B_shape, in_dtype),
            C: T.Buffer((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):

            A_shared = T.alloc_shared(A_shared_shape, in_dtype, scope=shared_scope)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype, scope=shared_scope)
            C_shared = T.alloc_shared(C_shared_shape, out_dtype, scope=shared_scope)
            A_local = T.alloc_local((warp_rows * local_size_a), in_dtype)
            B_local = T.alloc_local((warp_cols * local_size_b), in_dtype)
            C_local = T.alloc_local((warp_rows * warp_cols * local_size_c), accum_dtype)

            # T.annotate_layout({
            #     A_shared: make_swizzle_layout(A_shared),
            #     B_shared: make_swizzle_layout(B_shared),
            # })

            # # Improve L2 Cache
            # T.use_swizzle(panel_size=10)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=stage):

                # Load A into shared memory
                for i, k in T.Parallel(block_M, block_K):
                    A_shared[i, k] = A[by * block_M + i, ko * block_K + k]

                # Load B into shared memory
                for j, k in T.Parallel(block_N, block_K):
                    B_shared[j, k] = B[bx * block_N + j, ko * block_K + k]

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
                C[by * block_M + i, bx * block_N + j] = C_shared[
                    i // micro_size_x,
                    j // micro_size_y,
                    i % micro_size_x,
                    j % micro_size_y,
                ]

    return main


def assert_tl_w4a4_matmul_correctness(M, N, K, in_dtype, out_dtype, accum_dtype):
    matmul = tl_w4a4_matmul(M, N, K, in_dtype, out_dtype, accum_dtype)
    mod, params = TL.lower(matmul)
    src_code = mod.imported_modules[0].get_source()
    # src_code is the generated cuda source
    assert src_code is not None
    print(src_code)

    A = torch.randint(0, 4, (M, K), device="cuda", dtype=getattr(torch, in_dtype))
    B = torch.randint(0, 4, (N, K), device="cuda", dtype=getattr(torch, in_dtype))
    C = torch.zeros(M, N, device="cuda", dtype=getattr(torch, accum_dtype))

    compressed_A = (A[:, ::2] & 0x0F) + ((A[:, 1::2] & 0x0F) << 4)
    compressed_B = (B[:, ::2] & 0x0F) + ((B[:, 1::2] & 0x0F) << 4)
    mod = TL.Profiler(mod, params, [], TL.TensorSupplyType.Integer)
    mod(compressed_A, compressed_B, C)
    print(C.shape)
    print(C)
    # latency = mod.do_bench(mod.func, warmup=25, profiler="tvm")
    # print(latency)
    # # Ensure that the latency is not None
    # assert latency is not None

    # Get Reference Result
    ref_c = torch.matmul(A.to(torch.float32), B.T.to(torch.float32)).to(getattr(torch, accum_dtype))
    print(ref_c.shape)

    print(ref_c)
    torch.testing.assert_close(C, ref_c, rtol=1e-2, atol=1e-2)

def test_assert_tl_matmul_correctness():
    assert_tl_w4a4_matmul_correctness(16, 128, 64, "int8", "int32", "int32")


if __name__ == "__main__":
    test_assert_tl_matmul_correctness()
