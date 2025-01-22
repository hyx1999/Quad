# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import bitblas
import bitblas.testing

torch.random.manual_seed(42)

@torch.no_grad()
def matmul_int4_torch_forward():
    N = 128
    M = 128
    K = 128
    A_dtype = "int4"
    W_dtype = "int4"
    accum_dtype = "int32"
    out_dtype = "int32"
    layout = "nt"
    matmul_config = bitblas.MatmulConfig(
        M=M,  # M dimension
        N=N,  # N dimension
        K=K,  # K dimension
        A_dtype=A_dtype,  # activation A dtype
        W_dtype=W_dtype,  # weight W dtype
        accum_dtype=accum_dtype,  # accumulation dtype
        out_dtype=out_dtype,  # output dtype
        layout=layout,  # matrix layout, "nt" indicates the layout of A is non-transpose and the layout of W is transpose
        propagate_b=False,  # propagate B matrix
        fast_decoding=False,
    )

    matmul = bitblas.Matmul(config=matmul_config, enable_tuning=False)

    storage_dtype = "int8"
    print(getattr(torch, storage_dtype))
    A = torch.randint(0, 4, (M, K), device="cuda", dtype=getattr(torch, storage_dtype))
    C = torch.zeros(M, N, device="cuda", dtype=getattr(torch, out_dtype))
    B = torch.randint(0, 4, (N, K), device="cuda", dtype=getattr(torch, storage_dtype))
    compressed_A = (A[:, ::2] & 0x0F) + ((A[:, 1::2] & 0x0F) << 4)
    compressed_B = (B[:, ::2] & 0x0F) + ((B[:, 1::2] & 0x0F) << 4)
    matmul(compressed_A, compressed_B, output=C)

    print(C)
    
    # Get Reference Result
    ref_c = torch.matmul(A.to(torch.float32), B.T.to(torch.float32)).to(getattr(torch, out_dtype))
    print(ref_c)

def test_matmul_torch_forward():
    matmul_int4_torch_forward()

# fmt: on
if __name__ == "__main__":
    test_matmul_torch_forward()
