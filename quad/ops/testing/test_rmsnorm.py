import torch
import math
import tilelang as tl
import tilelang.language as T
import tilelang.testing
import tvm

# def rmsnorm(M, N, blk_m, blk_k):
#     dtype = "float"

#     @T.prim_func
#     def main(A: T.Buffer((M, N), dtype), B: T.Buffer((M, N), dtype)):
#         with T.Kernel(T.ceildiv(M, blk_m), threads=128) as bx:
#             A_shared = T.alloc_shared((blk_m, blk_k), dtype)
#             A_local = T.alloc_fragment((blk_m, blk_k), dtype)
#             A_powsum = T.alloc_fragment((blk_m,), dtype)

#             num_k_step = T.ceildiv(N, blk_k)
#             T.clear(A_local)
#             for k in range(num_k_step):
#                 T.copy(A[bx * blk_m, k * blk_k], A_shared)
#                 for i, j in T.Parallel(blk_m, blk_k):
#                     A_local[i, j] += A_shared[i, j] * A_shared[i, j]
#             T.reduce_sum(A_local, A_powsum, dim=1)
#             for i in T.Parallel(blk_m):
#                 A_powsum[i] = T.rsqrt(A_powsum[i] / N) + 1e-12

#             for k in range(num_k_step):
#                 # reverse, better cache hit rate
#                 T.copy(A[bx * blk_m, (num_k_step - 1 - k) * blk_k], A_shared)
#                 for i, j in T.Parallel(blk_m, blk_k):
#                     A_shared[i, j] *= A_powsum[i]
#                 T.copy(A_shared, B[bx * blk_m, (num_k_step - 1 - k) * blk_k])

#     return main


# def test_rms_norm(M = 16, N = 4096, blk_M = 1, blk_N = 512):
#     def ref_program(x):
#         return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-12)
    
#     program = rmsnorm(M, N, blk_M, blk_N)
#     # mod, params = tilelang.lower(program)
#     # print(rt_mod.imported_modules[0].get_source())
#     kernel = tilelang.compile(program, out_idx=-1, target="cuda", execution_backend="cython")
#     profiler = kernel.get_profiler()
#     print("entry.")
#     profiler.assert_allclose(ref_program, rtol=0.01, atol=0.01)
#     print("All checks pass.")
    
#     latency = profiler.do_bench(ref_program, warmup=500)
#     print("Ref: {:.2f} ms".format(latency))
#     latency = profiler.do_bench(profiler.mod, warmup=500)
#     print("Tile-lang: {:.2f} ms".format(latency))
    

# def fuse_rmsnorm_quant_fp16_to_int8(M, N, blk_m, blk_n, clip_ratio: float = 1.0, dtype="float32"):
#     assert blk_m == 1
#     num_threads = 128
#     min_int8 = -128
#     max_int8 = 127 

#     @T.prim_func
#     def main(
#         A: T.Buffer((M, N), dtype),
#         B_quant: T.Buffer((M, N), "int8"),
#         B_scale: T.Buffer((M,), dtype),
#     ):
#         with T.Kernel(T.ceildiv(M, blk_m), threads=num_threads) as bx:
#             A_shared = T.alloc_shared([blk_m, blk_n], dtype=dtype)
#             B_q_shared = T.alloc_shared([blk_m, blk_n], dtype="int8")
#             A_ma_blk = T.alloc_fragment([blk_m, blk_n], dtype="float32")  # ma -> max + abs
#             A_ps_blk = T.alloc_fragment([blk_m, blk_n], dtype="float32")  # ps -> pow + sum
#             A_ps = T.alloc_fragment([blk_m], dtype="float32")
#             A_ma = T.alloc_fragment([blk_m], dtype=dtype)
#             B_s_blk = T.alloc_fragment([blk_m], dtype="float32")
#             tid = T.get_thread_binding()
            
#             num_k_step = T.ceildiv(N, blk_n)
            
#             T.fill(A_ps_blk, 0.0)
#             T.fill(A_ma_blk, -T.infinity(dtype))

#             for k in T.Pipelined(num_k_step):
#                 T.copy(A[bx * blk_m, k * blk_n], A_shared)
#                 for i, j in T.Parallel(blk_m, blk_n):
#                     A_ps_blk[i, j] += A_shared[i, j] * A_shared[i, j]
#                     A_ma_blk[i, j] = T.max(A_ma_blk[i, j], T.abs(A_shared[i, j]))

#             T.reduce_sum(A_ps_blk, A_ps, dim=1)
#             T.reduce_max(A_ma_blk, A_ma, dim=1)

#             for i in T.serial(blk_m):
#                 A_ps[i] = T.rsqrt(A_ps[i] / N + 1e-12)  # powsum -> rstd
#                 B_s_blk[i] = (A_ma[i] * A_ps[i] / max_int8) * clip_ratio

#             for k in T.Pipelined(num_k_step):
#                 T.copy(A[bx * blk_m, (num_k_step - 1 - k) * blk_n], A_shared)
#                 for i, j in T.Parallel(blk_m, blk_n):
#                     B_q_shared[i, j] = T.clamp(
#                         T.cast(T.round(A_shared[i, j] * A_ps[i] / B_s_blk[i]), "int8"), 
#                         T.cast(min_int8, "int8"), 
#                         T.cast(max_int8, "int8")
#                     )
#                 T.copy(B_q_shared, B_quant[bx * blk_m, (num_k_step - 1 - k) * blk_n])

#             for i in T.serial(blk_m):
#                 if tid == 0:
#                     B_scale[bx * blk_m + i] = B_s_blk[i]

#     return main

# def test_rmsnorm_quant_fp16_to_int8(M = 16, N = 1024, blk_M = 1, blk_N = 512):

#     def quant_tensor(x: torch.Tensor, min_int: int = -128, max_int: int = 127):
#         x_scale = x.abs().max(dim=-1, keepdim=True).values / max_int
#         x_quant = torch.clamp(torch.round(x / x_scale), min_int, max_int).to(torch.int8)
#         return x_quant, x_scale

#     def ref_program(x):
#         return quant_tensor(x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-12))
    
#     program = fuse_rmsnorm_quant_fp16_to_int8(M, N, blk_M, blk_N)
#     # mod, params = tilelang.lower(program)
#     # print(rt_mod.imported_modules[0].get_source())
#     kernel = tilelang.compile(program, out_idx=[1, 2], target="cuda", execution_backend="cython")
#     print(kernel.get_kernel_source())
    
#     profiler = kernel.get_profiler()
#     print("entry.")
#     profiler.assert_allclose(ref_program, rtol=0.01, atol=0.01)
#     print("All checks pass.")
    
#     latency = profiler.do_bench(ref_program, warmup=500)
#     print("Ref: {:.2f} ms".format(latency))
#     latency = profiler.do_bench(profiler.mod, warmup=500)
#     print("Tile-lang: {:.2f} ms".format(latency))


def rmsnorm_fuse_quant_fp16_to_int4(M, N, R, blk_m, blk_n, dtype="float16"):
    num_threads = 128
    min_int4 = -8
    max_int4 = 7
    clip_ratio = 1.0
    assert R <= blk_n

    @T.prim_func
    def main(
        A: T.Buffer((M, R + N), dtype),
        B_r: T.Buffer((M, R), dtype),
        B_quant: T.Buffer((M, N // 2), "int8"),
        B_scale: T.Buffer((M,), dtype),
    ):
        with T.Kernel(T.ceildiv(M, blk_m), threads=num_threads) as bx:
            A_shared = T.alloc_shared([blk_m, blk_n], dtype=dtype)
            B_q_shared = T.alloc_shared([blk_m, blk_n // 2], dtype="uint8")
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
                A_ps[i] = T.rsqrt(A_ps[i] / (N + R) + 1e-12)  # powsum -> rstd
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
                    B_q_shared[i, j] = q_i8
                T.copy(B_q_shared, B_quant[bx * blk_m, (num_k_step - 1 - k) * (blk_n // 2)])

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

def test_rmsnorm_quant_fp16_to_int4(M = 1, N = 128, R = 64, blk_M = 1, blk_N = 128):
    from quad.ops.quant import sym_quant_int4
    def quant_tensor(x: torch.Tensor, min_int: int = -8, max_int: int = 7):
        x_scale = x.abs().max(dim=-1, keepdim=True).values / max_int
        x_quant_i4 = sym_quant_int4(x, x_scale).to(torch.int8)
        return x_quant_i4, x_scale

    def ref_program(x: torch.Tensor):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-12)
        x_r = x[..., :R].contiguous()
        x = x[..., R:].contiguous()
        return (x_r,) + quant_tensor(x)
    
    program = rmsnorm_fuse_quant_fp16_to_int4(M, N, R, blk_M, blk_N, dtype="float16")
    kernel = tilelang.compile(program, out_idx=[1, 2, 3], target="cuda", execution_backend="cython")
    kernel_source = kernel.get_kernel_source()
    print(kernel_source)
        
    profiler = kernel.get_profiler()
    print("entry.")
    profiler.assert_allclose(ref_program, rtol=0.01, atol=0.01)
    print("All checks pass.")
    
    latency = profiler.do_bench(ref_program, warmup=500)
    print("Ref: {:.2f} ms".format(latency))
    latency = profiler.do_bench(profiler.mod, warmup=500)
    print("Tile-lang: {:.2f} ms".format(latency))


if __name__ == "__main__":
    # test_rms_norm()
    # test_rmsnorm_quant_fp16_to_int8()
    test_rmsnorm_quant_fp16_to_int4()
