#pragma once

#include "cuda_prefix_sum/internal/kernel_array.hpp"
#include "cuda_prefix_sum/internal/kernel_launch_params.hpp"

__global__ void MultiTileKernel(
    KernelLaunchParams params,
    KernelArrayView right_tile_edges_buffer,
    KernelArrayView bottom_tile_edges_buffer
);