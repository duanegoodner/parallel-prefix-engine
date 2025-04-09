// cuda_prefix_sum_solver.cuh
#pragma once
#include <cuda_runtime.h>

#include "cuda_prefix_sum/kernel_launch_params.hpp"

void LaunchPrefixSumKernel(
    int *d_data,
    KernelLaunchParams kernel_params,
    // int full_matrix_dim_x,
    // int full_matrix_dim_y,
    // int tile_dim_x,
    // int tile_dim_y,
    cudaStream_t stream
);


