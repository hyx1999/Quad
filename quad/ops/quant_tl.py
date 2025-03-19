import torch
import math
import tilelang as tl
import tilelang.language as T
from collections import defaultdict
from typing import Tuple

kernel_cache = defaultdict(lambda: None)

def tl_quant_i4(M, N, clip_ratio: float, strides: Tuple[int, ...]):
    blk_m = 1 
    blk_n = 512
    num_threads = 128
    min_int4 = -8
    max_int4 = 7 
    in_dtype = "float16"
    out_dtype = "int8"
    scale_dtype = "float16"

    @T.prim_func
    def main(
        A: T.Buffer((M, N), in_dtype),
        B: T.Buffer((M, N // 2), out_dtype),
        B_scale: T.Buffer((M,), scale_dtype),
    ):
        with T.Kernel(T.ceildiv(M, blk_m), threads=num_threads) as bx:
            A_shared = T.alloc_shared([blk_m, blk_n], dtype=in_dtype, scope="shared")
            B_shared = T.alloc_fragment([blk_m, blk_n // 2], dtype=out_dtype)
            A_ma_blk = T.alloc_fragment([blk_m, blk_n], dtype="float32")  # ma -> max + abs
            A_ma = T.alloc_fragment([blk_m], dtype="float32")
            B_s = T.alloc_fragment([blk_m], dtype="float32")
            tid = T.get_thread_binding()
            
            T.annotate_layout({
                A: T.Layout(
                    (M, N), lambda i, j: [i * strides[0] + j * strides[1]]
                )
            })
            
            num_k_step = T.ceildiv(N, blk_n)

            T.fill(A_ma_blk, -T.infinity("float32"))
            for k in T.Pipelined(num_k_step):
                T.copy(A[bx * blk_m, k * blk_n], A_shared)
                for i, j in T.Parallel(blk_m, blk_n):
                    A_ma_blk[i, j] = T.max(A_ma_blk[i, j], T.abs(A_shared[i, j]))
            
            T.reduce_max(A_ma_blk, A_ma, dim=1)

            for i in T.serial(blk_m):
                B_s[i] = (A_ma[i] / max_int4) * clip_ratio

            for k in T.Pipelined(num_k_step):
                T.copy(A[bx * blk_m, k * blk_n], A_shared)
                for i, j in T.Parallel(blk_m, blk_n // 2):
                    j0 = 2 * j
                    j1 = 2 * j + 1
                    q0_ = T.cast(T.clamp(
                        T.cast(T.round(A_shared[i, j0] / B_s[i]), "int32"), 
                        min_int4, 
                        max_int4
                    ), "int8")
                    q1_ = T.cast(T.clamp(
                        T.cast(T.round(A_shared[i, j1] / B_s[i]), "int32"),
                        min_int4, 
                        max_int4,
                    ), "int8")
                    q0 = T.if_then_else(q0_ < 0, q0_ + T.cast(2 ** 4, "int8"), q0_)
                    q1 = T.if_then_else(q1_ < 0, q1_ + T.cast(2 ** 4, "int8"), q1_)
                    B_shared[i, j] = q0 | (q1 << 4)
                T.copy(B_shared, B[bx * blk_m, k * (blk_n // 2)])

            if tid == 0:
                for i in T.serial(blk_m):
                    B_scale[bx * blk_m + i] = B_s[i]

    return main


def get_quant_i4_kernel(hidden_size: int, clip_ratio: float, strides: Tuple[int, ...]):
    if kernel_cache[(hidden_size, clip_ratio, "i4")] is None:
        print("init tl_quant_i4 kernel...")
        program = tl_quant_i4(
            T.symbolic("num_tokens"),
            hidden_size,
            clip_ratio,
            strides
        )        
        quant_i4_kernel = tl.compile(program, out_idx=[1, 2], target="cuda", execution_backend="cython")
        kernel_cache[(hidden_size, clip_ratio, "i4")] = quant_i4_kernel
    else:
        quant_i4_kernel = kernel_cache[(hidden_size, clip_ratio, "i4")]

    return quant_i4_kernel
    # @torch.no_grad()
    # def quant_i4(x, clip_ratio: float):
    #     return quant_i4_kernel(x, clip_ratio=clip_ratio)
    

def tl_quant_i8(M, N, clip_ratio: float, strides: Tuple[int, ...]):
    blk_m = 1 
    blk_n = 512
    num_threads = 128
    min_int8 = -128
    max_int8 = 127
    in_dtype = "float16"
    out_dtype = "int8"
    scale_dtype = "float16"

    @T.prim_func
    def main(
        A: T.Buffer((M, N), in_dtype),
        B: T.Buffer((M, N), out_dtype),
        B_scale: T.Buffer((M,), scale_dtype),
    ):
        with T.Kernel(T.ceildiv(M, blk_m), threads=num_threads) as bx:
            A_shared = T.alloc_shared([blk_m, blk_n], dtype=in_dtype, scope="shared")
            B_shared = T.alloc_fragment([blk_m, blk_n], dtype=out_dtype)
            A_ma_blk = T.alloc_fragment([blk_m, blk_n], dtype="float32")  # ma -> max + abs
            A_ma = T.alloc_fragment([blk_m], dtype="float32")
            B_s_blk = T.alloc_fragment([blk_m], dtype="float32")
            tid = T.get_thread_binding()

            T.annotate_layout({
                A: T.Layout(
                    (M, N), lambda i, j: [i * strides[0] + j * strides[1]]
                )
            })

            num_k_step = T.ceildiv(N, blk_n)

            T.fill(A_ma_blk, -T.infinity("float32"))
            for k in T.Pipelined(num_k_step):
                T.copy(A[bx * blk_m, k * blk_n], A_shared)
                for i, j in T.Parallel(blk_m, blk_n):
                    A_ma_blk[i, j] = T.max(A_ma_blk[i, j], T.abs(A_shared[i, j]))
            
            T.reduce_max(A_ma_blk, A_ma, dim=1)

            for i in T.serial(blk_m):
                B_s_blk[i] = (A_ma[i] / max_int8) * clip_ratio

            for k in T.Pipelined(num_k_step):
                T.copy(A[bx * blk_m, k * blk_n], A_shared)
                for i, j in T.Parallel(blk_m, blk_n):
                    B_shared[i, j] = T.cast(T.clamp(
                        T.round(A_shared[i, j] / B_s_blk[i]), 
                        min_int8, 
                        max_int8,
                    ), "int8")
                T.copy(B_shared, B[bx * blk_m, k * blk_n])

            if tid == 0:
                for i in T.serial(blk_m):
                    B_scale[bx * blk_m + i] = B_s_blk[i]

    return main


def get_quant_i8_kernel(hidden_size: int, clip_ratio: float, strides: Tuple[int, ...]):
    if kernel_cache[(hidden_size, clip_ratio, "i8")] is None:
        print("init tl_quant_i8 kernel...")
        program = tl_quant_i8(
            T.symbolic("num_tokens"),
            hidden_size,
            clip_ratio,
            strides
        )
        quant_i8_kernel = tl.compile(program, out_idx=[1, 2], target="cuda", execution_backend="cython")
        kernel_cache[(hidden_size, clip_ratio, "i8")] = quant_i8_kernel
    else:
        quant_i8_kernel = kernel_cache[(hidden_size, clip_ratio, "i8")]
    
    return quant_i8_kernel
    # @torch.no_grad()
    # def quant_i8(x, clip_ratio: float):
    #     return quant_i8_kernel(x, clip_ratio=clip_ratio)
