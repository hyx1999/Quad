import quad
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.parameter import Parameter
from typing import Optional
from collections import namedtuple

QuantParams = namedtuple('QuantParams', ['clip_ratio', 'act_dtype'])

class QuantFn(Function):
    
    @torch.compile
    @staticmethod
    def forward(ctx, x: torch.Tensor, params: QuantParams):
        if params.act_dtype == "int4":
            scales_x: torch.Tensor = (torch.max(torch.abs(x), dim=-1)\
                .values.unsqueeze(-1) / 7)\
                .to(x.dtype) * params.clip_ratio
            q_min = -8
            q_max = 7
        else:
            scales_x: torch.Tensor = (torch.max(torch.abs(x), dim=-1)\
                .values.unsqueeze(-1) / 127)\
                .to(x.dtype) * params.clip_ratio
            q_min = -128
            q_max = 127
        eps = 1e-5
        scales_x[scales_x == 0] = eps
        quantized_x = (x / scales_x).round().clamp(q_min, q_max)
        x = quantized_x.type(x.dtype) * scales_x
        return x
    
    @staticmethod
    def backward(ctx, grad_x):
        if not ctx.needs_input_grad[0]:
            grad_x = None
        return grad_x, None


class TunableQuantizer(torch.nn.Module):  # Quantizier with STE backward pass

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
    
    def split(self, x: torch.Tensor):
        outlier_x, x = torch.split(x, [self.pod_rank, self.hidden_size], dim=-1) \
            if self.pod_rank != 0 else (None, x)
        return x, outlier_x
    
    def quantize(self, x: torch.Tensor):
        return QuantFn.apply(
            x, QuantParams(clip_ratio=self.input_clip_ratio, act_dtype=self.act_dtype)
        )

    def forward(self, x):
        x, outlier_x = self.split(x)
        x = self.quantize(x)
        return quad.TensorPack(x, outlier_x)


class TunableIdentity(torch.nn.Module):

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
    
    def split(self, x: torch.Tensor):
        outlier_x, x = torch.split(x, [self.pod_rank, self.hidden_size], dim=-1) \
            if self.pod_rank != 0 else (None, x)
        return x, outlier_x

    def forward(self, x):
        x, outlier_x = self.split(x)
        return quad.TensorPack(x, outlier_x)
