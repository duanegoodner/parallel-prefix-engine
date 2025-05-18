#pragma once

#include "cuda_prefix_sum/internal/kernel_launch_params.hpp"

__global__ void MultiTileKernel(
    KernelLaunchParams params,
    int *right_tile_edges_buffer,
    int *bottom_tile_edges_buffer
);