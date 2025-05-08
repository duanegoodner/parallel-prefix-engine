// prefix_sum_kernel.cuh
#pragma once

#include "cuda_prefix_sum/kernel_launch_params.hpp"

__global__ void SubtileKernel(KernelLaunchParams params);
