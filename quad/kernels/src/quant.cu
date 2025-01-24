#include <quant.h>

template <typename T>
__device__ __half int_to_half(T value)
{
    return __int2half_rn(static_cast<int>(value));
}

template <typename T>
__device__ float int_to_float(T value)
{
    return __int2float_rn(static_cast<float>(value));
}

template <typename T>
__device__ __half float_to_half(T value)
{
    return __float2half(static_cast<float>(value));
}

template <typename T>
__device__ __nv_bfloat16 float_to_bf16(T value)
{
    return __float2bfloat16(static_cast<float>(value));
}

__global__ void sym_quantize_f16_i4_kernel(
    const half *__restrict__ x,
    const half *__restrict__ scale,
    uint32_t rows,
    uint32_t colsSrc,
    uint32_t colsDst,
    uint32_t stride_row,
    Int4Storage *__restrict__ q)
{
    uint32_t row = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t colDst = threadIdx.x + blockIdx.x * blockDim.x;
    if (row >= rows || colDst * kElementsPerVector >= colsSrc)
    {
        return;
    }
    Int4Storage storage;
    memset(&storage, 0, sizeof(storage));
    uint32_t id = colDst * kElementsPerVector + row * stride_row;
#pragma unroll
    for (int i = 0; i < kElementsPerVector; ++i)
    {
        bool safe = (colDst * kElementsPerVector + i) < colsSrc;
        if (safe)
        {
            half data = __hdiv(x[id + i], scale[row]);

            int qval = clamp(__half2int_rn(data), qmin, qmax);
            Int4Subbyte{reinterpret_cast<cutlass::int4b_t *>(&storage), i}.set(
                qval);
        }
    }

    q[colDst + row * colsDst] = storage;
}

void sym_quant_fp16_i4_host(
    const half *x,
    const half *scale,
    uint32_t rows,
    uint32_t colsSrc,
    uint32_t colsDst,
    uint32_t stride_row,
    Int4Storage *q)
{

    dim3 block{std::min<uint32_t>(colsDst, 32), std::min<uint32_t>(rows, 16)};
    dim3 grid{cdiv(colsDst, block.x), cdiv(rows, block.y)};
    sym_quantize_f16_i4_kernel<<<grid, block>>>(x, scale, rows, colsSrc, colsDst, stride_row, q);
}

__global__ void sym_quantize_f16_i8_kernel(
    const half *__restrict__ x,
    const half *__restrict__ scale,
    uint32_t rows,
    uint32_t cols,
    uint32_t stride_row,
    int8_t *__restrict__ q)
{
    uint32_t row = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t colDst = threadIdx.x + blockIdx.x * blockDim.x;
    if (row >= rows || colDst >= cols)
    {
        return;
    }
    int8_t storage;
    memset(&storage, 0, sizeof(storage));
    uint32_t id = colDst + row * stride_row;
    bool safe = colDst < cols;
    if (safe)
    {
        half data = __hdiv(x[id], scale[row]);
        storage = static_cast<int8_t>(clamp(__half2int_rn(data), qmin_int8, qmax_int8));
    }
    q[colDst + row * cols] = storage;
}

void sym_quant_fp16_i8_host(
    const half *x,
    const half *scale,
    uint32_t rows,
    uint32_t cols,
    uint32_t stride_row,
    int8_t *q)
{

    dim3 block{std::min<uint32_t>(cols, 32), std::min<uint32_t>(rows, 16)};
    dim3 grid{cdiv(cols, block.x), cdiv(rows, block.y)};
    sym_quantize_f16_i8_kernel<<<grid, block>>>(x, scale, rows, cols, stride_row, q);
}

__global__ void sym_dequantize_i32_f16_kernel(
    const int32_t *__restrict__ q,
    const half *__restrict__ scale_row,
    const half *__restrict__ scale_col,
    uint32_t rows, uint32_t cols,
    half *__restrict__ x)
{
    uint32_t row = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t col = threadIdx.x + blockIdx.x * blockDim.x;

    if (col >= cols || row >= rows)
    {
        return;
    }

    float xElement = int_to_float(q[col + row * cols]);
    float scale_r = __half2float(scale_row[row]);
    float scale_c = __half2float(scale_col[col]);
    x[col + row * cols] = float_to_half(scale_r * scale_c * xElement);
}

void sym_dequant_host(const int32_t *q,
                      const half *scale_row,
                      const half *scale_col,
                      uint32_t rows,
                      uint32_t cols,
                      half *x)
{
    dim3 block{std::min<uint32_t>(cols, 16), std::min<uint32_t>(rows, 16)};
    dim3 grid{cdiv(cols, block.x), cdiv(rows, block.y)};
    sym_dequantize_i32_f16_kernel<<<grid, block>>>(
        q,
        scale_row, scale_col,
        rows, cols, x);
}

__global__ void sym_dequantize_i4_fp16_kernel(
    const Int4Storage *__restrict__ q,
    const half *__restrict__ scale_row,
    uint32_t rows, uint32_t colsSrc, uint32_t colsDst,
    half *__restrict__ x)
{
    uint32_t row = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row >= rows || col >= colsSrc)
    {
        return;
    }
    Int4Storage storage = q[col + row * colsSrc];
    uint32_t id = col * kElementsPerVector + row * colsDst;
#pragma unroll
    for (int i = 0; i < kElementsPerVector; ++i)
    {
        bool safe = (col * kElementsPerVector + i) < colsDst;
        if (safe)
        {
            int q_val = cutlass::int4b_t::xint_t(
                Int4Subbyte{reinterpret_cast<cutlass::int4b_t *>(&storage), i}.get());
            float scale_r = __half2float(scale_row[row]);
            x[id + i] = float_to_half(scale_r * q_val);
        }
    }
}

void sym_dequant_fp16_weight_host(const Int4Storage *q,
                             const half *scale_row,
                             uint32_t rows,
                             uint32_t colsSrc,
                             uint32_t colsDst,
                             half *x)
{
    dim3 block{std::min<uint32_t>(colsSrc, 16), std::min<uint32_t>(rows, 16)};
    dim3 grid{cdiv(colsSrc, block.x), cdiv(rows, block.y)};
    sym_dequantize_i4_fp16_kernel<<<grid, block>>>(
        q,
        scale_row,
        rows, colsSrc, colsDst, x);
}

__global__ void sym_dequantize_i4_bf16_kernel(
    const Int4Storage *__restrict__ q,
    const __nv_bfloat16 *__restrict__ scale_row,
    uint32_t rows, uint32_t colsSrc, uint32_t colsDst,
    __nv_bfloat16 *__restrict__ x)
{
    uint32_t row = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row >= rows || col >= colsSrc)
    {
        return;
    }
    Int4Storage storage = q[col + row * colsSrc];
    uint32_t id = col * kElementsPerVector + row * colsDst;
#pragma unroll
    for (int i = 0; i < kElementsPerVector; ++i)
    {
        bool safe = (col * kElementsPerVector + i) < colsDst;
        if (safe)
        {
            int q_val = cutlass::int4b_t::xint_t(
                Int4Subbyte{reinterpret_cast<cutlass::int4b_t *>(&storage), i}.get());
            float scale_r = __bfloat162float(scale_row[row]);
            x[id + i] = float_to_bf16(scale_r * q_val);
        }
    }
}

void sym_dequant_bf16_weight_host(const Int4Storage *q,
                             const __nv_bfloat16 *scale_row,
                             uint32_t rows,
                             uint32_t colsSrc,
                             uint32_t colsDst,
                             __nv_bfloat16 *x)
{
    dim3 block{std::min<uint32_t>(colsSrc, 16), std::min<uint32_t>(rows, 16)};
    dim3 grid{cdiv(colsSrc, block.x), cdiv(rows, block.y)};
    sym_dequantize_i4_bf16_kernel<<<grid, block>>>(
        q,
        scale_row,
        rows, colsSrc, colsDst, x);
}
