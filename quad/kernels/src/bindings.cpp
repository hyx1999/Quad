#include <torch/extension.h>

// Include all files
#include <quant.h>
#include <gemm.h>

torch::Tensor matmul_w4a4(const torch::Tensor &A, const torch::Tensor &B)
{
  torch::checkAllContiguous("matmul", {{A, "A", 0},
                                       {B, "B", 1}});
  torch::checkDeviceType("matmul", {A, B}, at::DeviceType::CUDA);

  torch::checkAllSameGPU("matmul", {{A, "A", 0},
                                    {B, "B", 1}});
  uint32_t M = A.size(0);
  uint32_t N = B.size(0);
  uint32_t K = A.size(1) * kElementsPerVector; // 4bit packing is on the columns
  auto C = torch::empty({M, N}, torch::dtype(torch::kInt32).device(A.device()));

  matmul_w4a4_host(A.data_ptr<Int4Storage>(), B.data_ptr<Int4Storage>(), M, N, K, C.data_ptr<int32_t>());

  return C;
}

torch::Tensor matmul_w4a8(const torch::Tensor &A, const torch::Tensor &B)
{
  torch::checkAllContiguous("matmul", {{A, "A", 0},
                                       {B, "B", 1}});
  torch::checkDeviceType("matmul", {A, B}, at::DeviceType::CUDA);

  torch::checkAllSameGPU("matmul", {{A, "A", 0},
                                    {B, "B", 1}});
  uint32_t M = A.size(0);
  uint32_t N = B.size(0);
  uint32_t K = A.size(1);
  auto C = torch::empty({M, N}, torch::dtype(torch::kInt32).device(A.device()));

  matmul_w4a8_host(A.data_ptr<int8_t>(), B.data_ptr<Int4Storage>(), M, N, K, C.data_ptr<int32_t>());

  return C;
}

/*
torch::Tensor matmul_w8a8(const torch::Tensor &A, const torch::Tensor &B)
{
  torch::checkAllContiguous("matmul", {{A, "A", 0},
                                       {B, "B", 1}});
  torch::checkDeviceType("matmul", {A, B}, at::DeviceType::CUDA);

  torch::checkAllSameGPU("matmul", {{A, "A", 0},
                                    {B, "B", 1}});
  uint32_t M = A.size(0);
  uint32_t N = B.size(0);
  uint32_t K = A.size(1); // 4bit packing is on the columns
  auto C = torch::empty({M, N}, torch::dtype(torch::kInt32).device(A.device()));

  matmul_w8a8_host(A.data_ptr<int8_t>(), B.data_ptr<int8_t>(), M, N, K, C.data_ptr<int32_t>());

  return C;
}
*/

torch::Tensor sym_quant_fp16_i4(const torch::Tensor &x, const torch::Tensor &scale)
{
  torch::checkAllContiguous("sym_quant", {{scale, "scale", 1}});
  torch::checkDeviceType("sym_quant", {x, scale}, at::DeviceType::CUDA);

  torch::checkSameGPU("sym_quant", {x, "x", 0}, {scale, "scale", 1});
  torch::checkSize("sym_quant", torch::TensorArg{scale, "scale", 1}, 0, x.size(0));
  uint32_t rows = x.size(0);
  uint32_t colsSrc = x.size(1);
  uint32_t colsDst = cdiv(colsSrc, kElementsPerVector);
  uint32_t stride_row = x.stride(0);
  uint32_t stride_col = x.stride(1);
  TORCH_CHECK(stride_col == 1);

  auto q = torch::empty({rows, colsDst}, torch::dtype(torch::kUInt8).device(x.device()));

  sym_quant_fp16_i4_host((half *)x.data_ptr(), (half *)scale.data_ptr(), rows, colsSrc, colsDst, stride_row, q.data_ptr<Int4Storage>());

  return q;
}

torch::Tensor sym_quant_fp16_i8(const torch::Tensor &x, const torch::Tensor &scale)
{
  torch::checkAllContiguous("sym_quant", {{scale, "scale", 1}});
  torch::checkDeviceType("sym_quant", {x, scale}, at::DeviceType::CUDA);

  torch::checkSameGPU("sym_quant", {x, "x", 0}, {scale, "scale", 1});
  torch::checkSize("sym_quant", torch::TensorArg{scale, "scale", 1}, 0, x.size(0));
  uint32_t rows = x.size(0);
  uint32_t cols = x.size(1);
  uint32_t stride_row = x.stride(0);
  uint32_t stride_col = x.stride(1);
  TORCH_CHECK(stride_col == 1);

  auto q = torch::empty({rows, cols}, torch::dtype(torch::kInt8).device(x.device()));

  sym_quant_fp16_i8_host((half *)x.data_ptr(), (half *)scale.data_ptr(), rows, cols, stride_row, q.data_ptr<int8_t>());

  return q;
}

torch::Tensor sym_dequant(const torch::Tensor &q,
                          const torch::Tensor &scale_row,
                          const torch::Tensor &scale_col,
                          const int bits)
{
  torch::checkAllContiguous("sym_dequant",
                            {{q, "q", 0},
                             {scale_row, "scale_row", 1},
                             {scale_col, "scale_col", 2}});
  torch::checkDeviceType("sym_dequant", {q, scale_row, scale_col},
                         at::DeviceType::CUDA);

  torch::checkAllSameGPU("sym_dequant",
                         {{q, "q", 0},
                          {scale_row, "scale_row", 1},
                          {scale_col, "scale_col", 2}});

  uint32_t rows = q.size(0);
  uint32_t cols = q.size(1);

  torch::checkSize("sym_dequant", torch::TensorArg{scale_row, "scale_row", 1}, 0,
                   rows);
  torch::checkSize("sym_dequant", torch::TensorArg{scale_col, "scale_col", 2}, 0,
                   cols);

  auto x = torch::empty(q.sizes(), torch::dtype(torch::kHalf).device(q.device()));

  switch (bits)
  {
  case 32:
    sym_dequant_host(q.data_ptr<int32_t>(), (half *)scale_row.data_ptr(), (half *)scale_col.data_ptr(),
                     rows, cols, (half *)x.data_ptr());
    break;
  default:
    TORCH_CHECK(false, "Unsupported data type")
  }

  return x;
}

torch::Tensor sym_dequant_fp16_weight(const torch::Tensor &q,
                          const torch::Tensor &scale_row,
                          const int bits)
{
  torch::checkAllContiguous("sym_dequant_fp16_weight",
                            {{q, "q", 0},
                             {scale_row, "scale_row", 1}});
  torch::checkDeviceType("sym_dequant_fp16_weight", {q, scale_row},
                         at::DeviceType::CUDA);

  torch::checkAllSameGPU("sym_dequant_fp16_weight",
                         {{q, "q", 0},
                          {scale_row, "scale_row", 1}});

  uint32_t rows = q.size(0);
  uint32_t colsSrc = q.size(1);
  uint32_t colsDst = colsSrc * kElementsPerVector;

  torch::checkSize("sym_dequant", torch::TensorArg{scale_row, "scale_row", 1}, 0,
                   rows);

  auto x = torch::empty({rows, colsDst}, torch::dtype(torch::kHalf).device(q.device()));

  switch (bits)
  {
  case 4:
    sym_dequant_fp16_weight_host(q.data_ptr<Int4Storage>(), (half *)scale_row.data_ptr(),
                     rows, colsSrc, colsDst, (half *)x.data_ptr());
    break;
  default:
    TORCH_CHECK(false, "Unsupported data type")
  }

  return x;
}


torch::Tensor sym_dequant_bf16_weight(const torch::Tensor &q,
                          const torch::Tensor &scale_row,
                          const int bits)
{
  torch::checkAllContiguous("sym_dequant_bf16_weight",
                            {{q, "q", 0},
                             {scale_row, "scale_row", 1}});
  torch::checkDeviceType("sym_dequant_bf16_weight", {q, scale_row},
                         at::DeviceType::CUDA);

  torch::checkAllSameGPU("sym_dequant_bf16_weight",
                         {{q, "q", 0},
                          {scale_row, "scale_row", 1}});

  uint32_t rows = q.size(0);
  uint32_t colsSrc = q.size(1);
  uint32_t colsDst = colsSrc * kElementsPerVector;

  torch::checkSize("sym_dequant", torch::TensorArg{scale_row, "scale_row", 1}, 0,
                   rows);

  auto x = torch::empty({rows, colsDst}, torch::dtype(torch::kBFloat16).device(q.device()));

  switch (bits)
  {
  case 4:
    sym_dequant_bf16_weight_host(q.data_ptr<Int4Storage>(), (__nv_bfloat16 *)scale_row.data_ptr(),
                     rows, colsSrc, colsDst, (__nv_bfloat16 *)x.data_ptr());
    break;
  default:
    TORCH_CHECK(false, "Unsupported data type")
  }

  return x;
}

torch::Tensor sym_dequant_weight(const torch::Tensor &q,
                          const torch::Tensor &scale_row,
                          const int bits)
{
    if (scale_row.dtype() == torch::kHalf)
        return sym_dequant_fp16_weight(q, scale_row, bits);
    else if (scale_row.dtype() == torch::kBFloat16)
        return sym_dequant_bf16_weight(q, scale_row, bits);
    else
        TORCH_CHECK(false, "Unsupported data type");
}

//====== pybind ======

#define DEFINE_pybind(name) m.def(#name, &name, #name);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("matmul_w4a4", &matmul_w4a4,
        "input: (A: torch.Tensor(M x K, UINT8, CUDA), B: torch.Tensor(N x K, "
        "UINT8, CUDA))\n"
        "output: torch.Tensor(M x N, INT32, CUDA)\n"
        "output = int4Unpacking(A) @ int4Unpacking(B)^T",
        py::arg("A"), py::arg("B"));
  m.def("matmul_w4a8", &matmul_w4a8,
        "input: (A: torch.Tensor(M x K, INT8, CUDA), B: torch.Tensor(N x K, "
        "UINT8, CUDA))\n"
        "output: torch.Tensor(M x N, INT32, CUDA)\n"
        "output = A @ int4Unpacking(B)^T",
        py::arg("A"), py::arg("B"));

  m.def("sym_quant_fp16_i4", &sym_quant_fp16_i4,
        "input: (src: torch.Tensor(M x N, FP16, CUDA), scale: "
        "torch.Tensor(M x 1, FP16, CUDA))\n"
        "output: torch.Tensor(M x ceil(N / 2), UINT8, CUDA)\n"
        "output = int4Packing(int4Rounding(source / scale)\n",
        py::arg("x"), py::arg("scale"));
  m.def("sym_quant_fp16_i8", &sym_quant_fp16_i8,
        "input: (src: torch.Tensor(M x N, FP16, CUDA), scale: "
        "torch.Tensor(M x 1, FP16, CUDA))\n"
        "output: torch.Tensor(M x N, INT8, CUDA)\n"
        "output = int8(Rounding(source / scale)\n",
        py::arg("x"), py::arg("scale"));

  m.def("sym_dequant", &sym_dequant,
        "input (x: torch.Tensor(M x N), scale_row: torch.Tensor(M x 1, "
        "FP16), scale_col: torch.Tensor(1 x N, FP16)"
        "bits: int\n"
        "output: torch.Tensor(M x N, FP16)\n"
        "output = x * scale_row * scale_col"
        "when bits equal 8: "
        "input x type is int8\n"
        "when bits equal 16: "
        "input x type is FP16\n"
        "when bits equal 32: "
        "input x type is int32\n",
        py::arg("q"), py::arg("scale_row"), py::arg("scale_col"),
        py::arg("bits"));
m.def("sym_dequant_weight", &sym_dequant_weight,
        "input: (src: torch.Tensor(M x N / 2, UINT8, CUDA), scale: "
        "torch.Tensor(M x 1, FP16/BF16, CUDA))\n"
        "output: torch.Tensor(M x N, FP16/BF16, CUDA)\n",
        py::arg("q"), py::arg("scale_row"),
        py::arg("bits"));
}
