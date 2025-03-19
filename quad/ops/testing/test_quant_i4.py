import torch
import math
import tilelang as tl
import tilelang.language as T
import tilelang.testing
import quad

torch.manual_seed(0)

def tl_quant_i4(M, N, clip_ratio: float):
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
                B_s_blk[i] = (A_ma[i] / max_int4) * clip_ratio

            for k in T.Pipelined(num_k_step):
                T.copy(A[bx * blk_m, k * blk_n], A_shared)
                for i, j in T.Parallel(blk_m, blk_n // 2):
                    j0 = 2 * j
                    j1 = 2 * j + 1
                    q0_ = T.cast(T.clamp(
                        T.cast(T.round(A_shared[i, j0] / B_s_blk[i]), "int32"), 
                        min_int4, 
                        max_int4
                    ), "int8")
                    q1_ = T.cast(T.clamp(
                        T.cast(T.round(A_shared[i, j1] / B_s_blk[i]), "int32"),
                        min_int4, 
                        max_int4,
                    ), "int8")
                    q0 = T.if_then_else(q0_ < 0, q0_ + T.cast(2 ** 4, "int8"), q0_)
                    q1 = T.if_then_else(q1_ < 0, q1_ + T.cast(2 ** 4, "int8"), q1_)
                    B_shared[i, j] = q0 | (q1 << 4)
                T.copy(B_shared, B[bx * blk_m, k * (blk_n // 2)])

            if tid == 0:
                for i in T.serial(blk_m):
                    B_scale[bx * blk_m + i] = B_s_blk[i]

    return main

def get_scales(x: torch.Tensor, q_max: float, input_clip_ratio: float = 1.0):
    scales_x = (torch.max(torch.abs(x), dim=-1).values.unsqueeze(-1) / q_max).to(
        torch.float16
    ) * input_clip_ratio
    return scales_x

def quantize_int4(x: torch.Tensor):
    scales_x = get_scales(x, 7)
    quantized_x = quad.ops.sym_quant_int4(x, scales_x).to(torch.int8)
    return quantized_x, scales_x.view(-1)

def test_packi4(M = 2048, N = 3584):
    program = tl_quant_i4(T.symbolic("m"), N, 1.0)
    kernel = tilelang.compile(program, out_idx=[1, 2], target="cuda", execution_backend="cython")
    kernel_source = kernel.get_kernel_source()
    print(kernel_source)
    
    def ref_program(x):
        return quantize_int4(x)
    
    x = torch.randn((M, N), dtype=torch.float16, device="cuda")
    x_quant, x_scale = kernel(x)
    print(x_quant)
    print(x_scale)
        
    x_quant_ref, x_scale_ref = ref_program(x)
    print(x_quant_ref)
    print(x_scale_ref)
        
    # torch.testing.assert_close(x_quant, x_quant_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(x_scale, x_scale_ref, rtol=1e-2, atol=1e-2)


if __name__ == '__main__':
    test_packi4()
