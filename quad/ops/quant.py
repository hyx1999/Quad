import torch
import quad_cuda
from .utils import flatten_last_dim_and_return_shape

# def sym_quant_int8(x: torch.Tensor, scale: torch.Tensor, num_bits=8):
    # Qn = -(2**(num_bits - 1))
    # Qp = 2**(num_bits - 1) - 1
    # result = (x / scale).round().clamp(Qn, Qp)
    # return result.type(torch.int8)

def sym_quant_int8(x: torch.Tensor, scale: torch.Tensor):
    assert x.dtype == scale.dtype == torch.float16
    x, x_shape_excl_last = flatten_last_dim_and_return_shape(x)
    return quad_cuda.sym_quant_fp16_i8(x, scale.view(-1)).view(*x_shape_excl_last, -1)

def sym_quant_int4(x, scale):
    assert x.dtype == scale.dtype == torch.float16
    x, x_shape_excl_last = flatten_last_dim_and_return_shape(x)
    return quad_cuda.sym_quant_fp16_i4(x, scale.view(-1)).view(*x_shape_excl_last, -1)

def sym_dequant(q, scale_row, scale_col, bits=32):
    assert q.dtype == torch.int32
    assert scale_row.dtype == scale_col.dtype == torch.float16
    q, q_shape_excl_last = flatten_last_dim_and_return_shape(q)
    return quad_cuda.sym_dequant(q, scale_row.view(-1), scale_col, bits).view(*q_shape_excl_last, -1)

def sym_dequant_weight(q, scale_row, bits=4):
    assert q.dtype == torch.uint8
    assert scale_row.dtype == torch.float16 or scale_row.dtype == torch.bfloat16
    q, q_shape_excl_last = flatten_last_dim_and_return_shape(q)
    return quad_cuda.sym_dequant_weight(q, scale_row.view(-1), bits).view(*q_shape_excl_last, -1)
