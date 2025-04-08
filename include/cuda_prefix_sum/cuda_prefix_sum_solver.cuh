// cuda_prefix_sum_solver.cuh
#pragma once
#include <cuda_runtime.h>


void LaunchPrefixSumKernel(
    int *d_data,
    int full_matrix_dim_x,
    int full_matrix_dim_y,
    int tile_dim_x,
    int tile_dim_y,
    cudaStream_t stream
);
