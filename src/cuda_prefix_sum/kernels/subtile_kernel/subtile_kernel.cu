// cuda_prefix_sum_solver.cu
//
// Defines the CUDA kernel and launch function for performing 2D prefix sum.
// This file contains only GPU-side logic and is compiled by NVCC.

#include <cuda_runtime.h>

#include <cstdio>
#include <iostream>

#include "cuda_prefix_sum/internal/subtile_device_helpers.cuh"
#include "cuda_prefix_sum/internal/kernel_config_utils.cuh"
#include "cuda_prefix_sum/internal/kernel_launch_params.hpp"
#include "cuda_prefix_sum/subtile_kernel_launcher.cuh"

__global__ void SubtileKernel(
    // int *d_data,
    KernelLaunchParams params
) {
  // Declare dynamic shared memory
  extern __shared__ int shared_mem[];

  // Declare shared memory
  KernelArray array_a{.d_address = shared_mem, .size = params.tile_size};
  __syncthreads();

  // === Phase 1: Load input from global memory to shared memory ===
  CopyFromGlobalToShared(params.array, array_a, params.sub_tile_size);
  __syncthreads();
  
  // === Phase 2: Compute 2D prefix sum on shared mem array ===
  ComputeSharedMemArrayPrefixSum(array_a, params.sub_tile_size);
  __syncthreads();

  // === Phase 3: Write final result back to global memory ===
  CopyFromSharedToGlobal(array_a, params.array, params.sub_tile_size);
}
