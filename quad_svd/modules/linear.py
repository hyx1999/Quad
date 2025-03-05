import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import quad_cuda
from torch.nn.parameter import Parameter
from typing import Optional
import quad

class QuantLinearW4A4(torch.nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        pod_rank: int = 0,
        svd_rank: int = 0,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            "weight_scales",
            torch.zeros((self.out_features, 1), requires_grad=False, device=device),
        )
        self.register_buffer(
            "weight",
            (
                torch.zeros(
                    (self.out_features, self.in_features // 2),
                    # SubByte weight
                    dtype=torch.uint8,
                    device=device,
                    requires_grad=False,
                )
            ),
        )
        if bias:
            self.register_buffer("bias", torch.zeros((self.out_features), dtype=dtype))
        else:
            self.bias = None
        if pod_rank > 0 and svd_rank > 0:
            raise ValueError("pod_rank > 0 and svd_rank > 0")
        if pod_rank > 0:
            self.w_outlier = Parameter(
                torch.empty((out_features, pod_rank), **factory_kwargs)
            )
        else:
            self.register_parameter("w_outlier", None)
        if svd_rank > 0:
            self.w_outlier = Parameter(
                torch.empty((out_features, pod_rank), **factory_kwargs)
            )
        else:
            self.register_parameter("w_outlier", None)
        self.matmul = self.init_matmul()

    def init_matmul(self):
        return lambda A, W: quad_cuda.matmul_w4a4(A, W)  # A @ W.T

    def forward(self, x_pack):
        quant_x, x_shape = quad.ops.flatten_last_dim_and_return_shape(x_pack.x.quantized_x)
        scale_x = x_pack.x.scales_x.view(-1)
        x_out = self.matmul(quant_x, self.weight)
        x_out = quad.ops.sym_dequant(x_out, scale_x, self.weight_scales)
        if self.w_outlier is not None:
            outlier_x, _ = quad.ops.flatten_last_dim_and_return_shape(x_pack.outlier_x)
            x_out.addmm_(outlier_x, self.w_outlier.T)
        if self.bias is not None:
            x_out.add_(self.bias)
        return x_out.view(x_shape + (-1,))

    @staticmethod
    def from_float(
        module: torch.nn.Linear,
        extra_in=0,
        extra_out=0,
        pod_rank: int = 0,
        svd_rank: int = 0,
    ):
        """
        Generate a new Linear4bit module from a FP16 Linear module.
        The weight matrix should have the same shape as the weight matrix of the FP16 Linear module and rounded using torch.round()
        routine. We will convert it to subByte representation and save it in the int_weight buffer.
        """
        weight_matrix = module.weight.data

        packed_module = QuantLinearW4A4(
            module.in_features + extra_in,
            module.out_features + extra_out,
            bias=module.bias is not None,
            dtype=weight_matrix.dtype,
            device=weight_matrix.device,
            pod_rank=pod_rank,
            svd_rank=svd_rank,
        )
        return packed_module


class QuantLinearW4A8(QuantLinearW4A4):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor
    
    def __init__(self, in_features, out_features, bias = True, device=None, dtype=None, pod_rank: int = 0, svd_rank: int = 0):
        super().__init__(in_features, out_features, bias, device, dtype, pod_rank, svd_rank)

    def init_matmul(self):
        return lambda A, W: quad_cuda.matmul_w4a8(A, W)  # A @ W.T

    @staticmethod
    def from_float(
        module: torch.nn.Linear,
        extra_in=0,
        extra_out=0,
        pod_rank: int = 0,
        svd_rank: int = 0,
    ):
        """
        Generate a new Linear4bit module from a FP16 Linear module.
        The weight matrix should have the same shape as the weight matrix of the FP16 Linear module and rounded using torch.round()
        routine. We will convert it to subByte representation and save it in the int_weight buffer.
        """
        weight_matrix = module.weight.data

        packed_module = QuantLinearW4A8(
            module.in_features + extra_in,
            module.out_features + extra_out,
            bias=module.bias is not None,
            dtype=weight_matrix.dtype,
            device=weight_matrix.device,
            pod_rank=pod_rank,
            svd_rank=svd_rank,
        )
        return packed_module
