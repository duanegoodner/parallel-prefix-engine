// prefix_sum_kernel.cuh
#pragma once

#include "cuda_prefix_sum/internal/kernel_launch_params.hpp"

__global__ void SingleTileKernel(KernelLaunchParams params);
