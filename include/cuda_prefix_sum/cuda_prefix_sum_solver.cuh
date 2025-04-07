// cuda_prefix_sum_solver.cuh
#pragma once
#include <cuda_runtime.h>

void LaunchPrefixSumKernel(
    int *d_data,
    int full_matrix_dim_x,
    int full_matrix_dim_y,
    // int rows_per_block,
    // int cols_per_block,
    // int blocks_per_row,
    // int blocks_per_col,
    cudaStream_t stream
);
