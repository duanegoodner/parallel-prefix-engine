#pragma once

#include "common/array_size_2d.hpp"

__global__ void RowWiseScanSingleBlock(
    const int *__restrict__ in,
    int *__restrict__ out,
    // int num_cols
    ArraySize2D size
);

__global__ void RowWiseScanMultiBlockPhase1(
    const int* __restrict__ in,
    int* __restrict__ out,
    int* __restrict__ block_sums,
    // int num_cols,
    ArraySize2D size,
    int chunk_size);


__global__ void RowWiseScanMultiBlockPhase2(
    int* __restrict__ out,
    const int* __restrict__ scanned_block_sums,
    // int num_cols,
    ArraySize2D size,
    int chunk_size);