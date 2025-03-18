import torch
import math
import tilelang as tl
import tilelang.language as T
from collections import defaultdict

kernel_cache = defaultdict(lambda: None)

def tl_rmsnorm_fuse_quant_fp16_to_int4(M, N, R, clip_ratio, eps, dtype="float16"):
    blk_m = 1 
    blk_n = 512
    num_threads = 128
    min_int4 = -8
    max_int4 = 7
    assert R <= blk_n

    @T.prim_func
    def main(
        A: T.Buffer((M, R + N), dtype),
        B_r: T.Buffer((M, R), dtype),
        B_quant: T.Buffer((M, N // 2), "int8"),
        B_scale: T.Buffer((M,), dtype),
    ):
        with T.Kernel(T.ceildiv(M, blk_m), threads=num_threads) as bx:
            A_shared = T.alloc_shared([blk_m, blk_n], dtype="float32")
            B_shared = T.alloc_shared([blk_m, blk_n // 2], dtype="uint8")
            A_ma_blk = T.alloc_fragment([blk_m, blk_n], dtype="float32")  # ma -> max + abs
            A_ps_blk = T.alloc_fragment([blk_m, blk_n], dtype="float32")  # ps -> pow + sum
            A_ps = T.alloc_fragment([blk_m], dtype="float32")
            A_ma = T.alloc_fragment([blk_m], dtype="float32")
            B_s_blk = T.alloc_fragment([blk_m], dtype="float32")
            tid = T.get_thread_binding()
            
            num_k_step = T.ceildiv(N, blk_n)
            
            T.fill(A_ps_blk, 0.0)
            T.fill(A_ma_blk, -T.infinity(dtype))

            for k in T.Pipelined(num_k_step):
                T.copy(A[bx * blk_m, R + k * blk_n], A_shared)
                for i, j in T.Parallel(blk_m, blk_n):
                    A_ps_blk[i, j] += A_shared[i, j] * A_shared[i, j]
                    A_ma_blk[i, j] = T.max(A_ma_blk[i, j], T.abs(A_shared[i, j]))
            
            # R <= blk_n
            T.copy(A[bx * blk_m, 0], A_shared)
            for i, j in T.Parallel(blk_m, blk_n):
                if j < R:
                    A_ps_blk[i, j] += A_shared[i, j] * A_shared[i, j]

            T.reduce_sum(A_ps_blk, A_ps, dim=1)
            T.reduce_max(A_ma_blk, A_ma, dim=1)
            
            for i in T.serial(blk_m):
                A_ps[i] = T.rsqrt(A_ps[i] / (N + R) + eps)  # powsum -> rstd
                B_s_blk[i] = (A_ma[i] * A_ps[i] / max_int4) * clip_ratio

            for k in T.Pipelined(num_k_step):
                T.copy(A[bx * blk_m, R + (num_k_step - 1 - k) * blk_n], A_shared)
                for i, j in T.Parallel(blk_m, blk_n // 2):
                    j0 = 2 * j
                    j1 = 2 * j + 1
                    q0_ = T.clamp(
                        T.cast(T.round(A_shared[i, j0] * A_ps[i] / B_s_blk[i]), "int8"), 
                        T.cast(min_int4, "int8"), 
                        T.cast(max_int4, "int8")
                    )
                    q1_ = T.clamp(
                        T.cast(T.round(A_shared[i, j1] * A_ps[i] / B_s_blk[i]), "int8"), 
                        T.cast(min_int4, "int8"), 
                        T.cast(max_int4, "int8")
                    )
                    q0 = T.cast(T.if_then_else(q0_ < 0, q0_ + T.cast(2 ** 4, "int8"), q0_), "uint8")
                    q1 = T.cast(T.if_then_else(q1_ < 0, q1_ + T.cast(2 ** 4, "int8"), q1_), "uint8")
                    q_i8 = q0 | (q1 << 4)
                    B_shared[i, j] = q_i8
                T.copy(B_shared, B_quant[bx * blk_m, (num_k_step - 1 - k) * (blk_n // 2)])

            for i, j in T.Parallel(blk_m, blk_n):
                if j < R:
                    A_shared[i, j] = A[bx * blk_m + i, j]
            for i, j in T.Parallel(blk_m, blk_n):
                if j < R:
                    B_r[bx * blk_m + i, j] = A_shared[i, j] * A_ps[i]

            if tid == 0:
                for i in T.serial(blk_m):
                    B_scale[bx * blk_m + i] = B_s_blk[i]

    return main

def get_rmsnorm_fuse_quant_kernel(hidden_size: int, rank: int, clip_ratio: float, eps: float):
    if kernel_cache[(hidden_size, rank, clip_ratio, eps)] is None:
        print("init tl_rmsnorm_fuse_quant_fp16_to_int4 kernel...")
        program = tl_rmsnorm_fuse_quant_fp16_to_int4(
            T.symbolic("num_tokens"),
            hidden_size,
            rank,
            clip_ratio,
            eps
        )        
        rmsnorm_fuse_quant_kernel = tl.compile(program, out_idx=[1, 2, 3], target="cuda", execution_backend="cython")
        kernel_cache[(hidden_size, rank, clip_ratio, eps)] = rmsnorm_fuse_quant_kernel
    else:
        rmsnorm_fuse_quant_kernel = kernel_cache[(hidden_size, rank, clip_ratio, eps)]

    return rmsnorm_fuse_quant_kernel
    # @torch.no_grad()
    # def rmsnorm_fuse_quant(x, clip_ratio: float):
    #     return rmsnorm_fuse_quant_kernel(x, clip_ratio=clip_ratio)
