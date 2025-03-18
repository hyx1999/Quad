import torch
import tilelang as tl
import tilelang.language as T
# `make_mma_swizzle_layout` is a python-defined layout function
# that helps align data for MMA (Matrix Multiply-Accumulate) operations.
from tilelang.intrinsics import make_mma_swizzle_layout as make_swizzle_layout

def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
    @T.prim_func
    def main(
        A: T.Buffer((M, K), dtype),
        B: T.Buffer((K, N), dtype),
        C: T.Buffer((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            # Allocate shared and local fragments
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local  = T.alloc_fragment((block_M, block_N), accum_dtype)

            # Annotate memory layout
            T.annotate_layout({
                A_shared: make_swizzle_layout(A_shared),
                B_shared: make_swizzle_layout(B_shared),
            })

            # Enable swizzle-based rasterization for better L2 locality
            T.use_swizzle(panel_size=10, enable=True)

            # Clear the local accumulation buffer
            T.clear(C_local)

            # Pipelined iteration over K dimension
            for idx in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                # Copy tile of A
                T.copy(A[by * block_M, idx * block_K], A_shared)

                # Parallel copy tile of B
                for ko, j in T.Parallel(block_K, block_N):
                    B_shared[ko, j] = B[idx * block_K + ko, bx * block_N + j]

                # Perform local GEMM on the shared-memory tiles
                T.gemm(A_shared, B_shared, C_local)

            # Copy the result tile back
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main

def assert_tl_matmul_correctness(M, N, K, dtype, accum_dtype):
    program = matmul(M, N, K, 64, 64, 64, dtype, accum_dtype)
    mod, params = tl.lower(program)
    src_code = mod.imported_modules[0].get_source()
    print(src_code)

    # src_code is the generated cuda source
    assert src_code is not None
        
    matmul_kernel = tl.compile(program, target="cuda", execution_backend="cython")
    profiler = matmul_kernel.get_profiler()

    A = torch.randn((M, K), device="cuda", dtype=getattr(torch, dtype))
    B = torch.randint((N, K), device="cuda", dtype=getattr(torch, dtype))
    C = torch.zeros(M, N, device="cuda", dtype=getattr(torch, dtype))

    matmul_kernel(A, B, C)
    print(C)

    # Get Reference Result
    def ref_program():
        ref_c = torch.matmul(A, B)
        return ref_c

    ref_c = ref_program()
    print(ref_c)
    torch.testing.assert_close(C, ref_c, rtol=1e-2, atol=1e-2)
    
    latency = profiler.do_bench(profiler.mod, warmup=500)
    print("Tile-lang: {:.2f} ms".format(latency))


def test_assert_tl_matmul_correctness():
    assert_tl_matmul_correctness(128, 128, 128, "float16", "float32")


if __name__ == "__main__":
    test_assert_tl_matmul_correctness()
