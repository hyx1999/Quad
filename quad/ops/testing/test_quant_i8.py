import torch
import math
import tilelang as tl
import tilelang.language as T
import tilelang.testing
import tvm

def tl_quant_i8(M, N, clip_ratio: float):
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
                T.copy(A[bx * blk_m, k * blk_n], A_shared)  # cause "RuntimeError: CUDA error: misaligned address"
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


def test_packi8(M = 16, N = 1024):
    program = tl_quant_i8(M, N, 1.0)
    kernel = tilelang.compile(program, out_idx=[1, 2], target="cuda", execution_backend="cython")
    kernel_source = kernel.get_kernel_source()
    print(kernel_source)
    
    def ref_program(x):
        x_scale = x.abs().max(dim=-1, keepdim=True).values / 127
        x_quant = torch.round(x / x_scale).to(torch.int8)
        return x_quant, x_scale.view(-1)
    
    x = torch.randn((M, N), dtype=torch.float16, device="cuda")
    x_quant, x_scale = kernel(x)
    print(x_quant)
    print(x_scale)
    
    x_quant_ref, x_scale_ref = ref_program(x)
    print(x_quant_ref)
    print(x_scale_ref)

if __name__ == '__main__':
    test_packi8()

# prelude="""
# #include <cuda_runtime.h>
# #include <cutlass/fast_math.h>
# #include <cutlass/numeric_types.h>
# #include <math_constants.h>

# using cutlass::bfloat16_t;
# using cutlass::half_t;
# using cutlass::tfloat32_t;

# #define TL_DEVICE __forceinline__ __device__

# TL_DEVICE int make_int_tmp(signed char x0, signed char x1, signed char x2,
#                        signed char x3) {
#   return (x3 << 24) | (x2 << 16) | (x1 << 8) | x0;
# }

# using int2_t = int2;
# // Pack sixteen char values.
# TL_DEVICE int2_t make_int2(signed char x0, signed char x1, signed char x2,
#                            signed char x3, signed char y0, signed char y1,
#                            signed char y2, signed char y3) {
#   int2_t result;
#   result.x = make_int_tmp(x0, x1, x2, x3);
#   result.y = make_int_tmp(y0, y1, y2, y3);
#   return result;
# }
# """

# def tl_quant_i8(M, N, clip_ratio: float):
#     blk_m = 1 
#     blk_n = 512
#     num_threads = 128
#     min_int8 = -128
#     max_int8 = 127
#     in_dtype = "float16"
#     out_dtype = "int8"
#     scale_dtype = "float16"

#     @T.prim_func
#     def main(
#         A: T.Buffer((M, N), in_dtype),
#         B: T.Buffer((M, N), out_dtype),
#         B_scale: T.Buffer((M,), scale_dtype),
#     ):
#         with T.Kernel(T.ceildiv(M, blk_m), threads=num_threads, prelude=prelude) as bx:
#             A_shared = T.alloc_shared([blk_m, blk_n], dtype=in_dtype)
#             B_shared = T.alloc_shared([blk_m, blk_n], dtype=out_dtype)
#             A_ma_blk = T.alloc_fragment([blk_m, blk_n], dtype="float32")  # ma -> max + abs
#             A_ma = T.alloc_fragment([blk_m], dtype="float32")
#             B_s_blk = T.alloc_fragment([blk_m], dtype="float32")
#             tid = T.get_thread_binding()

#             num_k_step = T.ceildiv(N, blk_n)

#             T.fill(A_ma_blk, -T.infinity("float32"))
#             for k in T.Pipelined(num_k_step):
#                 T.copy(A[bx * blk_m, k * blk_n], A_shared)
#                 for i, j in T.Parallel(blk_m, blk_n):
#                     A_ma_blk[i, j] = T.max(A_ma_blk[i, j], T.abs(A_shared[i, j]))
            
#             T.reduce_max(A_ma_blk, A_ma, dim=1)

#             for i in T.serial(blk_m):
#                 B_s_blk[i] = (A_ma[i] / max_int8) * clip_ratio

#             for k in T.Pipelined(num_k_step):
#                 T.copy(A[bx * blk_m, (num_k_step - 1 - k) * blk_n], A_shared)
#                 for i, j in T.Parallel(blk_m, blk_n):
#                     B_shared[i, j] = T.clamp(
#                         T.cast(T.round(A_shared[i, j] / B_s_blk[i]), "int8"), 
#                         T.cast(min_int8, "int8"), 
#                         T.cast(max_int8, "int8")
#                     )
#                 T.copy(B_shared, B[bx * blk_m, (num_k_step - 1 - k) * blk_n])

#             if tid == 0:
#                 for i in T.serial(blk_m):
#                     B_scale[bx * blk_m + i] = B_s_blk[i]

#     return main


# def test_packi8(M = 16, N = 1024):
#     program = tl_quant_i8(M, N, 1.0)
#     kernel = tilelang.compile(program, out_idx=[1, 2], target="cuda", execution_backend="cython")
#     kernel_source = kernel.get_kernel_source()
#     print(kernel_source)
    
#     x = torch.randn((M, N), dtype=torch.float16, device="cuda")
#     x_pack, x_scale = kernel(x)
#     print(x_pack)
#     print(x_scale)

# if __name__ == '__main__':
#     test_packi8()