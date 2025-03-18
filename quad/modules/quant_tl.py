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
        input_clip_ratio: float = 1.0,
        act_dtype: str = "int4",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_clip_ratio = input_clip_ratio
        self.act_dtype = act_dtype
        if act_dtype == "int4":
            self.kernel = get_quant_i4_kernel(hidden_size, input_clip_ratio)
        elif act_dtype == "int8":
            self.kernel = get_quant_i8_kernel(hidden_size, input_clip_ratio)
        else:
            raise ValueError

    def forward(self, x):
        x, x_shape = flatten_last_dim_and_return_shape(x)
        x_quant, x_scale = self.kernel(x)
        return quad.TensorPack(
            quad.QTensor(
                x_quant.view(*x_shape, -1), 
                x_scale.view(*x_shape, -1),
            ), 
            None
        )
