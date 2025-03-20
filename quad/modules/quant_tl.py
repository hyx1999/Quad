import quad
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
        input_clip_ratio: float = 1.0,
        act_dtype: str = "int4",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.pod_rank = pod_rank
        self.input_clip_ratio = input_clip_ratio
        self.act_dtype = act_dtype
        if act_dtype == "int4":
            self.kernel = get_quant_i4_kernel(hidden_size, input_clip_ratio, (hidden_size, 1))
        elif act_dtype == "int8":
            self.kernel = get_quant_i8_kernel(hidden_size, input_clip_ratio, (hidden_size, 1))
        else:
            raise ValueError

    def split(self, x: torch.Tensor):
        outlier_x, x = torch.split(x, [self.pod_rank, self.hidden_size], dim=-1) \
            if self.pod_rank != 0 else (None, x)
        return x, outlier_x

    def quantize(self, x: torch.Tensor):
        x, x_shape = flatten_last_dim_and_return_shape(x.contiguous())
        x_quant, x_scale = self.kernel(x)
        return quad.QTensor(x_quant.view(*x_shape, -1), x_scale.view(*x_shape))

    def forward(self, x):
        x, outlier_x = self.split(x)
        x = self.quantize(x)
        return quad.TensorPack(x, outlier_x)
