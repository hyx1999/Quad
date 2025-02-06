import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import quad
from torch.nn.parameter import Parameter
from torch.autograd import Function
from typing import Optional


class QuantLinearFn(Function):        

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight_scales: torch.Tensor, qweight: torch.Tensor):
        ctx.save_for_backward(x, weight_scales, qweight)
        x, x_shape = quad.ops.flatten_last_dim_and_return_shape(x)
        weight = quad.ops.sym_dequant_weight(qweight, weight_scales)
        y = x @ weight.T
        return y.view(*x_shape, -1)
    
    @staticmethod
    def backward(ctx, grad_y):
        x, weight_scales, qweight = ctx.saved_tensors
        grad_y, _ = quad.ops.flatten_last_dim_and_return_shape(grad_y)
        x, x_shape = quad.ops.flatten_last_dim_and_return_shape(x)
        weight = quad.ops.sym_dequant_weight(qweight, weight_scales)
        qweight_f = quad.ops.sym_dequant_weight(qweight, torch.ones_like(weight_scales))
        grad_x = grad_scale = None
        if ctx.needs_input_grad[0]:
            grad_x = (grad_y @ weight).view(*x_shape, -1)
        if ctx.needs_input_grad[1]:
            grad_weight = (x.T @ grad_y).T
            grad_scale = (grad_weight * qweight_f).sum(dim=1, keepdim=True)
        return grad_x, grad_scale, None


class TunableQuantLinear(torch.nn.Module):
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
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_parameter(
            "weight_scales",
            Parameter(torch.zeros((self.out_features, 1), device=device)),
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
            self.register_parameter("bias", Parameter(torch.zeros((self.out_features), dtype=dtype)))
        else:
            self.bias = None
        if pod_rank > 0:
            self.register_parameter(
                "w_outlier",
                Parameter(torch.empty((out_features, pod_rank), **factory_kwargs))
            )
        else:
            self.register_parameter("w_outlier", None)
        
    def forward(self, x_pack):
        x_out = QuantLinearFn.apply(x_pack.x, self.weight_scales, self.weight)
        if self.w_outlier is not None:
            x_out = x_out + F.linear(x_pack.outlier_x, self.w_outlier)
        if self.bias is not None:
            x_out = x_out + self.bias
        return x_out

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

        packed_module = TunableQuantLinear(
            module.in_features + extra_in,
            module.out_features + extra_out,
            bias=module.bias is not None,
            dtype=weight_matrix.dtype,
            device=weight_matrix.device,
            pod_rank=pod_rank,
        )
        return packed_module
