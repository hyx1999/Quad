#pragma once

#include <common.h>


void sym_quant_fp16_i4_host(
        const half *x,
        const half *scale,
        uint32_t rows,
        uint32_t colsSrc,
        uint32_t colsDst,
        Int4Storage *q
);

void sym_quant_fp16_i8_host(
        const half *x,
        const half *scale,
        uint32_t rows,
        uint32_t cols,
        int8_t *q
);

void sym_dequant_host(
        const int32_t *q,
        const half *scale_row,
        const half *scale_col,
        uint32_t rows,
        uint32_t cols,
        half *x
);
