import torch
import bitblas
import quad_cuda

torch.random.manual_seed(42)

with torch.no_grad():
    N = 32
    M = 32
    K = 64
    A_dtype = "int8"
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
        propagate_a=False,
        propagate_b=False,
        fast_decoding=False,
    )
    matmul = bitblas.Matmul(config=matmul_config)

    # num_bits = 4
    # zeros = 2 ** (num_bits - 1)

    A = torch.randint(-127, 127 + 1, (M, K), dtype=torch.int8).cuda()
    B = torch.randint(-8, 7 + 1, (N, K), dtype=torch.int8).cuda()
    
    out_ref = A.type(torch.float32) @ B.type(torch.float32).T
    print(out_ref.type(torch.int32))

    B = matmul.transform_weight(B)    
    out1 = matmul(A, B)
    print(out1)
