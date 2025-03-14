import quad
import quad_svd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from typing import Optional


class Identity(torch.nn.Module):

    def __init__(self, hidden_size: int, pod_rank: int, svd_rank: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.pod_rank = pod_rank
        self.svd_rank = svd_rank
        if svd_rank > 0:
            self.lr_fc = nn.Linear(hidden_size, svd_rank, bias=False)
        else:
            self.register_module("lr_fc", None)
        assert svd_rank == 0 or pod_rank == 0

    def forward(self, x):
        if self.pod_rank > 0:
            outlier_x = x[..., : self.pod_rank].contiguous()
            x = x[..., self.pod_rank :].contiguous()
        elif self.svd_rank > 0:
            outlier_x = self.lr_fc(x)
        else:
            outlier_x = None
        return quad_svd.TensorPack(x, outlier_x)


class Quantizer(torch.nn.Module):

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
            self.quantize = self.quantize_int4
        else:
            self.quantize = self.quantize_int8
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
    
    def quantize_int4(self, x: torch.Tensor):
        scales_x = self.get_scales(x, 7)
        quantized_x = quad.ops.sym_quant_int4(x, scales_x)
        return quad_svd.QTensor(quantized_x, scales_x)

    def quantize_int8(self, x: torch.Tensor):
        scales_x = self.get_scales(x, 127)
        quantized_x = quad.ops.sym_quant_int8(x, scales_x)
        return quad_svd.QTensor(quantized_x, scales_x)

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
