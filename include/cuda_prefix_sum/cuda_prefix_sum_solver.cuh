// cuda_prefix_sum_solver.cuh
#pragma once
#include <cuda_runtime.h>

#include "cuda_prefix_sum/kernel_launch_params.hpp"

void LaunchPrefixSumKernelTiled(
    KernelLaunchParams kernel_params,
    cudaStream_t stream
);

void LaunchPrefixSumKernelSingleElement(
    KernelLaunchParams kernel_params,
    cudaStream_t cuda_stream
);

void LaunchPrefixSumKernelWarp(const int* d_input, int* d_output);
