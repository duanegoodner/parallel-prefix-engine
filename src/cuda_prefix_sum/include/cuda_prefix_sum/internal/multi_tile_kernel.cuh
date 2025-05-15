#pragma once

#include "cuda_prefix_sum/internal/kernel_launch_params.hpp"

__global__ void MultiBlockKernel(KernelLaunchParams params);