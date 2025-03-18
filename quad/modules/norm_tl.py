import torch
import quad
from quad.ops import (
    get_rmsnorm_fuse_quant_kernel,
    flatten_last_dim_and_return_shape,
)

class RMSNormFuseQuantTl(torch.nn.Module):

    def __init__(self, mean_dim: int, rank: int, clip_ratio: 1.0, eps=1e-5):
        super().__init__()
        self.mean_dim = mean_dim
        self.rank = rank
        self.clip_ratio = clip_ratio
        self.eps = eps
        self.kernel = get_rmsnorm_fuse_quant_kernel(mean_dim, rank, clip_ratio, eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, x_shape = flatten_last_dim_and_return_shape(x)
        x_r, x_quant, x_scale = self.kernel(x)
        x_r = x_r.view(*x_shape, -1)
        x_quant = x_quant.view(*x_shape, -1)
        x_scale = x_scale.view(*x_shape)
        return quad.TensorPack(quad.QTensor(x_quant, x_scale), x_r)
