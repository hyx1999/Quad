import torch
import math
import tilelang as tl
import tilelang.language as T
import tilelang.testing
import tvm

def tl_pack_i4(M, N, blk_m, blk_n):
    assert blk_m == 1
    assert blk_n % 2 == 0
    num_threads = 128
    min_int = -8
    max_int = 7  

    @T.prim_func
    def main(
        A: T.Buffer((M, N), "int8"),
        B: T.Buffer((M, N // 2), "uint8"),
    ):
        with T.Kernel(T.ceildiv(M, blk_m), threads=num_threads) as bx:
            A_shared = T.alloc_shared([blk_m, blk_n], dtype="int8", scope="shared")
            B_shared = T.alloc_fragment([blk_m, blk_n // 2], dtype="uint8")
            
            num_k_step = T.ceildiv(N, blk_n)
            
            for k in T.Pipelined(num_k_step):
                T.copy(A[bx * blk_m, k * blk_n], A_shared)  # cause "RuntimeError: CUDA error: misaligned address"
                for i, j in T.Parallel(blk_m, blk_n // 2):
                    j0 = 2 * j
                    j1 = 2 * j + 1
                    q0_ = T.clamp(
                        T.cast(T.round(A_shared[i, j0]), "int8"), 
                        T.cast(min_int, "int8"), 
                        T.cast(max_int, "int8")
                    )
                    q1_ = T.clamp(
                        T.cast(T.round(A_shared[i, j1]), "int8"), 
                        T.cast(min_int, "int8"), 
                        T.cast(max_int, "int8")
                    )
                    q0 = T.cast(T.if_then_else(q0_ < 0, q0_ + (2 ** 4), q0_), "uint8")
                    q1 = T.cast(T.if_then_else(q1_ < 0, q1_ + (2 ** 4), q1_), "uint8")
                    q_i8 = q0 | (q1 << 4)
                    B_shared[i, j] = q_i8
                T.copy(B_shared, B[bx * blk_m, k * (blk_n // 2)])

    return main

def test_packi4(M = 1, N = 16, blk_M = 1, blk_N = 16):
    program = tl_pack_i4(M, N, blk_M, blk_N)
    kernel = tilelang.compile(program, out_idx=[1], target="cuda", execution_backend="cython")
    kernel_source = kernel.get_kernel_source()
    print(kernel_source)
    
    x = torch.randint(-8, 7 + 1, (M, N), dtype=torch.int8, device="cuda")
    x_pack = kernel(x)
    print(x_pack)

if __name__ == '__main__':
    test_packi4()