#pragma once

#include <common.h>


void matmul_w4a4_host(
        const Int4Storage *A,
        const Int4Storage *B,
        uint32_t M,
        uint32_t N,
        uint32_t K,
        int32_t *C
);

void matmul_w4a8_host(
        const int8_t *A,
        const Int4Storage *B,
        uint32_t M,
        uint32_t N,
        uint32_t K,
        int32_t *C
);

/*
void matmul_w8a8_host(
        const int8_t *A,
        const int8_t *B,
        uint32_t M,
        uint32_t N,
        uint32_t K,
        int32_t *C
);
*/