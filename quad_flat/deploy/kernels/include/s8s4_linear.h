#pragma once
#include <torch/extension.h>

at::Tensor
s8s4_linear_cutlass(
    const at::Tensor& xq, const at::Tensor& x_scale, const at::Tensor& wq,
    const at::Tensor& w_scale, const at::Tensor& bias
);