from .hadamard import (
    matmul_hadU_cuda, 
    random_hadamard_matrix, 
    apply_exact_had_to_linear
)
from .quant import (
    sym_quant_int8,
    sym_quant_int4,
    sym_dequant,
    sym_dequant_weight,
)
from .utils import flatten_last_dim_and_return_shape
from .rmsnorm_tl import get_rmsnorm_fuse_quant_kernel
from .quant_tl import get_quant_i4_kernel, get_quant_i8_kernel
