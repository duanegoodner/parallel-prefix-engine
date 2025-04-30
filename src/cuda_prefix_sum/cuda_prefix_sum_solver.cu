// cuda_prefix_sum_solver.cu
//
// Defines the CUDA kernel and launch function for performing 2D prefix sum.
// This file contains only GPU-side logic and is compiled by NVCC.

#include <cuda_runtime.h>

#include <cstdio>
#include <iostream>

#include "cuda_prefix_sum/cuda_device_helpers.cuh"
#include "cuda_prefix_sum/cuda_prefix_sum_solver.cuh"
#include "cuda_prefix_sum/kernel_launch_params.hpp"

__global__ void PrefixSumKernel(
    // int *d_data,
    KernelLaunchParams params
) {
  // Declare dynamic shared memory
  extern __shared__ int shared_mem[];

  // Divide shared memory into two arrays
  KernelArray array_a{.d_address = shared_mem, .size = params.array.size};
  KernelArray array_b{
      .d_address = shared_mem + array_a.size.x * array_a.size.y,
      .size = params.array.size
  };

  // === Phase 1: Load input from global memory to shared memory ===
  CopyGlobalArrayToSharedArray(params.array, array_a, params.tile_size);

  __syncthreads();

  // === Phase 2: Row-wise prefix sum within each tile of arrayA ===
  for (int tile_col = 1; tile_col < params.tile_size.y; tile_col++) {
    for (int tile_row = 0; tile_row < params.tile_size.x; ++tile_row) {
      ComputeRowWisePrefixSum(array_a, params.tile_size, tile_row, tile_col);
    }
  }

  __syncthreads();

  // === Phase 3: Column-wise prefix sum within each tile of arrayA ===
  for (int tile_row = 1; tile_row < params.tile_size.x; tile_row++) {
    for (int tile_col = 0; tile_col < params.tile_size.y; ++tile_col) {
      ComputeColWisePrefixSum(array_a, params.tile_size, tile_row, tile_col);
    }
  }

  __syncthreads();

  // === Phase 3: Copy array_a to array_b ===

  CopySharedArrayToSharedArray(array_a, array_b, params.tile_size);
  __syncthreads();

  // === Phase 4: Broadcast array_a right edges to array_b

  BroadcastRightEdges(array_a, params.tile_size, array_b);
  __syncthreads();

  // === Phase 5: Copy array_b to array_a ===

  CopySharedArrayToSharedArray(array_b, array_a, params.tile_size);
  __syncthreads();


  // ==== Phase 6: Broadcast array_b bottom edges to array_a

  BroadcastBottomEdges(array_b, params.tile_size, array_a);
  __syncthreads();

  // === Phase 5: Write final result back to global memory ===
  CopySharedArrayToGlobalArray(array_a, params.array, params.tile_size);

  // PrintGlobalMemArray(d_data);
}

void LaunchPrefixSumKernel(
    // int *d_data,
    KernelLaunchParams kernel_params,
    cudaStream_t stream
) {

  int num_tiles_x = kernel_params.array.size.x / kernel_params.tile_size.x;
  int num_tiles_y = kernel_params.array.size.y / kernel_params.tile_size.y;

  // dim3 blockDim(full_matrix_dim_x, full_matrix_dim_y);
  dim3 blockDim(num_tiles_x, num_tiles_y);
  dim3 gridDim(1, 1); // Single block for now

  PrefixSumKernel<<<gridDim, blockDim, 0, stream>>>(
      // d_data,
      kernel_params
  );

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }

  cudaDeviceSynchronize();
}
