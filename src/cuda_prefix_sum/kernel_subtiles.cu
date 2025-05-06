// cuda_prefix_sum_solver.cu
//
// Defines the CUDA kernel and launch function for performing 2D prefix sum.
// This file contains only GPU-side logic and is compiled by NVCC.

#include <cuda_runtime.h>

#include <cstdio>
#include <iostream>

#include "cuda_prefix_sum/cuda_met_device_helpers.cuh"
#include "cuda_prefix_sum/cuda_prefix_sum_solver.cuh"
#include "cuda_prefix_sum/kernel_launch_params.hpp"

__global__ void PrefixSumKernelTiled(
    // int *d_data,
    KernelLaunchParams params
) {
  // Declare dynamic shared memory
  extern __shared__ int shared_mem[];

  // Divide shared memory into two arrays
  KernelArray array_a{.d_address = shared_mem, .size = params.array.size};
  KernelArray array_b{
      .d_address = shared_mem + array_a.size.num_rows * array_a.size.num_cols,
      .size = params.array.size
  };

  // === Phase 1: Load input from global memory to shared memory ===
  // CopyGlobalArrayToSharedArray(params.array, array_a, params.tile_size);
  CopyMETTiledArray(params.array, array_a, params.tile_size);
  __syncthreads();

  // === Phase 2: Row-wise prefix sum within each tile of arrayA ===
  // for (int tile_col = 1; tile_col < params.tile_size.y; tile_col++) {
  //   for (int tile_row = 0; tile_row < params.tile_size.x; ++tile_row) {
  //     ComputeRowWisePrefixSum(array_a, params.tile_size, tile_row,
  //     tile_col);
  //   }
  // }
  ComputeLocalRowWisePrefixSums(array_a, params.tile_size);
  __syncthreads();

  // === Phase 3: Column-wise prefix sum within each tile of arrayA ===
  // for (int tile_row = 1; tile_row < params.tile_size.x; tile_row++) {
  //   for (int tile_col = 0; tile_col < params.tile_size.y; ++tile_col) {
  //     ComputeColWisePrefixSum(array_a, params.tile_size, tile_row,
  //     tile_col);
  //   }
  // }
  ComputeLocalColWisePrefixSums(array_a, params.tile_size);
  __syncthreads();

  // === Phase 3: Copy array_a to array_b ===

  // CopySharedArrayToSharedArray(array_a, array_b, params.tile_size);
  CopyMETTiledArray(array_a, array_b, params.tile_size);
  __syncthreads();

  // === Phase 4: Broadcast array_a right edges to array_b

  BroadcastRightEdges(array_a, params.tile_size, array_b);
  __syncthreads();

  // === Phase 5: Copy array_b to array_a ===

  // CopySharedArrayToSharedArray(array_b, array_a, params.tile_size);
  CopyMETTiledArray(array_b, array_a, params.tile_size);
  __syncthreads();

  // ==== Phase 6: Broadcast array_b bottom edges to array_a

  BroadcastBottomEdges(array_b, params.tile_size, array_a);
  __syncthreads();

  // === Phase 5: Write final result back to global memory ===
  // CopySharedArrayToGlobalArray(array_a, params.array, params.tile_size);
  CopyMETTiledArray(array_a, params.array, params.tile_size);

  // PrintGlobalMemArray(d_data);
}

void LaunchPrefixSumKernelTiled(KernelLaunchParams kernel_params) {

  int num_tile_cols =
      kernel_params.array.size.num_cols / kernel_params.tile_size.num_cols;
  int num_tile_rows =
      kernel_params.array.size.num_rows / kernel_params.tile_size.num_rows;

  // dim3 blockDim(full_matrix_dim_x, full_matrix_dim_y);
  dim3 blockDim(num_tile_cols, num_tile_rows);
  dim3 gridDim(1, 1); // Single block for now

  int shared_mem_size = 2 * kernel_params.array.size.num_rows *
                        kernel_params.array.size.num_cols * sizeof(int);

  PrefixSumKernelTiled<<<gridDim, blockDim, shared_mem_size, 0>>>(kernel_params
  );

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }

  cudaDeviceSynchronize();
  cudaError_t sync_err = cudaGetLastError();
  if (sync_err != cudaSuccess) {
    fprintf(stderr, "CUDA sync error: %s\n", cudaGetErrorString(sync_err));
  }
}
