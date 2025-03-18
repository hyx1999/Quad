import quad_svd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from typing import Optional
from quad.ops import (
    get_quant_i4_kernel,
    get_quant_i8_kernel,
    flatten_last_dim_and_return_shape
)

class QuantizerTl(torch.nn.Module):

    def __init__(self, 
        hidden_size: int, 
        pod_rank: int, 
        svd_rank: int,
        input_clip_ratio: float = 1.0,
        act_dtype: str = "int4",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.pod_rank = pod_rank
        self.svd_rank = svd_rank
        self.input_clip_ratio = input_clip_ratio
        self.act_dtype = act_dtype
        if act_dtype == "int4":
            self.kernel = get_quant_i4_kernel(hidden_size, input_clip_ratio)
        else:
            self.kernel = get_quant_i8_kernel(hidden_size, input_clip_ratio)
        if svd_rank > 0:
            self.lr_fc = nn.Linear(hidden_size, svd_rank, bias=False)
        else:
            self.register_module("lr_fc", None)
        assert svd_rank == 0 or pod_rank == 0
    
    def split(self, x: torch.Tensor):
        outlier_x, x = torch.split(x, [self.pod_rank, self.hidden_size], dim=-1) \
            if self.pod_rank != 0 else (None, x)
        return x, outlier_x
    
    @torch.compile
    def get_scales(self, x: torch.Tensor, q_max: float):
        scales_x = (torch.max(torch.abs(x), dim=-1).values.unsqueeze(-1) / q_max).to(
            torch.float16
        ) * self.input_clip_ratio
        return scales_x
    
    def quantize(self, x: torch.Tensor):
        x, x_shape = flatten_last_dim_and_return_shape(x)
        x_quant, x_scale = self.kernel(x)
        return quad_svd.QTensor(x_quant.view(*x_shape, -1), x_scale.view(*x_shape))

    def forward(self, x):
        if self.pod_rank > 0:
            x, outlier_x = self.split(x)
        else:
            if self.lr_fc is not None:
                outlier_x = self.lr_fc(x)
            else:
                outlier_x = None
        x = self.quantize(x)
        return quad_svd.TensorPack(x, outlier_x)
