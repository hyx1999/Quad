import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import quad_cuda
from torch.nn.parameter import Parameter
from typing import Optional
import quad
from quad.ops import (
    get_fuse_w4a4_w16a16_matmul_kernel,
    get_fuse_w4a8_w16a16_matmul_kernel,
    get_w4a4_matmul_kernel,
    get_w4a8_matmul_kernel,
)

class QuantLinearW4A4Tl(torch.nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias=True,
        device=None,
        dtype=None,
        pod_rank=0,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            "weight_scales",
            torch.zeros((self.out_features, 1), requires_grad=False, **factory_kwargs),
        )
        self.register_buffer(
            "weight",
            (
                torch.zeros(
                    (self.out_features, self.in_features // 2),
                    # SubByte weight
                    dtype=torch.int8,
                    device=device,
                    requires_grad=False,
                )
            ),
        )
        self.register_buffer(
            "bias", 
            torch.zeros((self.out_features), requires_grad=False, **factory_kwargs)
        )
        if pod_rank > 0:
            self.register_buffer(
                "w_outlier", 
                torch.empty((out_features, pod_rank), requires_grad=False, **factory_kwargs)
            )
        else:
            self.register_buffer("w_outlier", None)
        self.matmul = self.init_matmul(pod_rank)

    def init_matmul(self, pod_rank: int):
        if pod_rank == 0:
            return lambda x_quant, x_scale, w_quant, w_scale, w_bias, x_r, w_r: \
                quad_cuda.s4s4_linear_cutlass(x_quant, x_scale, w_quant, w_scale, w_bias)
        else:
            return lambda x_quant, x_scale, w_quant, w_scale, w_bias, x_r, w_r: \
                quad_cuda.s4s4_linear_cutlass(x_quant, x_scale, w_quant, w_scale, w_bias).addmm_(x_r, w_r.T)

    def forward(self, x_pack):
        x_quant, x_shape = quad.ops.flatten_last_dim_and_return_shape(x_pack.x.quantized_x)
        x_scale = x_pack.x.scales_x.view(-1)
        x_r, _ = quad.ops.flatten_last_dim_and_return_shape(x_pack.outlier_x)
        x_out = self.matmul(x_quant, x_scale, self.weight, self.weight_scales, self.bias, x_r, self.w_outlier)
        return x_out.view(x_shape + (-1,))

    @staticmethod
    def from_float(
        module: torch.nn.Linear,
        extra_in=0,
        extra_out=0,
        pod_rank: int = 0,
    ):
        """
        Generate a new Linear4bit module from a FP16 Linear module.
        The weight matrix should have the same shape as the weight matrix of the FP16 Linear module and rounded using torch.round()
        routine. We will convert it to subByte representation and save it in the int_weight buffer.
        """
        weight_matrix = module.weight.data

        packed_module = QuantLinearW4A4Tl(
            module.in_features + extra_in,
            module.out_features + extra_out,
            bias=module.bias is not None,
            dtype=weight_matrix.dtype,
            device=weight_matrix.device,
            pod_rank=pod_rank,
        )
        return packed_module


class QuantLinearW4A8Tl(QuantLinearW4A4Tl):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor
    
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, pod_rank=0):
        super().__init__(in_features, out_features, bias, device, dtype, pod_rank)

    def init_matmul(self, pod_rank: int):
        if pod_rank == 0:
            return lambda x_quant, x_scale, w_quant, w_scale, w_bias, x_r, w_r: \
                quad_cuda.s8s4_linear_cutlass(x_quant, x_scale, w_quant, w_scale, w_bias)
        else:
            return lambda x_quant, x_scale, w_quant, w_scale, w_bias, x_r, w_r: \
                quad_cuda.s8s4_linear_cutlass(x_quant, x_scale, w_quant, w_scale, w_bias).addmm_(x_r, w_r.T)

    @staticmethod
    def from_float(
        module: torch.nn.Linear,
        extra_in=0,
        extra_out=0,
        pod_rank: int = 0,
    ):
        """
        Generate a new Linear4bit module from a FP16 Linear module.
        The weight matrix should have the same shape as the weight matrix of the FP16 Linear module and rounded using torch.round()
        routine. We will convert it to subByte representation and save it in the int_weight buffer.
        """
        weight_matrix = module.weight.data

        packed_module = QuantLinearW4A8Tl(
            module.in_features + extra_in,
            module.out_features + extra_out,
            bias=module.bias is not None,
            dtype=weight_matrix.dtype,
            device=weight_matrix.device,
            pod_rank=pod_rank,
        )
        return packed_module
