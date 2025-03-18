import torch
import math
import tilelang as tl
import tilelang.language as T
import tilelang.testing

def tl_sum(M, N, R, blk_m, blk_n, dtype="float16"):
    num_threads = 128
    assert R <= blk_n

    @T.prim_func
    def main(
        A: T.Buffer((M, R + N), dtype),
        B: T.Buffer((M,), dtype),
    ):
        with T.Kernel(T.ceildiv(M, blk_m), threads=num_threads) as bx:
            A_shared = T.alloc_shared([blk_m, blk_n], dtype=dtype)
            A_ps_blk = T.alloc_fragment([blk_m, blk_n], dtype="float32")  # ps -> pow + sum
            A_ps = T.alloc_fragment([blk_m], dtype="float32")
            tid = T.get_thread_binding()
            
            num_k_step = T.ceildiv(N, blk_n)
            
            T.fill(A_ps_blk, 0.0)

            for k in T.Pipelined(num_k_step):
                T.copy(A[bx * blk_m, R + k * blk_n], A_shared)
                for i, j in T.Parallel(blk_m, blk_n):
                    A_ps_blk[i, j] += A_shared[i, j] * A_shared[i, j]
            
            # R <= blk_n
            T.copy(A[bx * blk_m, 0], A_shared)
            # for i, j in T.Parallel(blk_m, blk_n):
            #     if j < R:
            #         A_ps_blk[i, j] += A_shared[i, j] * A_shared[i, j]
            for i, j in T.Parallel(blk_m, R):
                A_ps_blk[i, j] += A_shared[i, j] * A_shared[i, j]

            T.reduce_sum(A_ps_blk, A_ps, dim=1)
            
            if tid == 0:
                for i in T.serial(blk_m):
                    B[bx * blk_m + i] = A_ps[i]

    return main

def test_kernel(M = 1, N = 128, R = 64, blk_M = 1, blk_N = 128):

    def ref_program(x: torch.Tensor):
        return x.pow(2).sum(dim=-1)
    
    program = tl_sum(M, N, R, blk_M, blk_N, dtype="float16")
    kernel = tilelang.compile(program, out_idx=[1], target="cuda", execution_backend="cython")
    kernel_source = kernel.get_kernel_source()
    print(kernel_source)
    
    x = torch.ones(M, N + R, dtype=torch.float16, device="cuda")
    y0 = kernel(x)
    y1 = ref_program(x)
    
    print(y0)
    print(y1)
    
    profiler = kernel.get_profiler()
    print("entry.")
    profiler.assert_allclose(ref_program, rtol=0.01, atol=0.01)
    print("All checks pass.")
    
    latency = profiler.do_bench(ref_program, warmup=500)
    print("Ref: {:.2f} ms".format(latency))
    latency = profiler.do_bench(profiler.mod, warmup=500)
    print("Tile-lang: {:.2f} ms".format(latency))


if __name__ == "__main__":
    test_kernel()
