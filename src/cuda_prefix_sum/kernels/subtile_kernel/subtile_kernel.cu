// cuda_prefix_sum_solver.cu
//
// Defines the CUDA kernel and launch function for performing 2D prefix sum.
// This file contains only GPU-side logic and is compiled by NVCC.

#include <cuda_runtime.h>

#include <cstdio>
#include <iostream>

#include "cuda_prefix_sum/cuda_met_device_helpers.cuh"
#include "cuda_prefix_sum/cuda_prefix_sum_solver.cuh"
#include "cuda_prefix_sum/internal/kernel_config_utils.cuh"
#include "cuda_prefix_sum/internal/kernel_launch_params.hpp"
#include "cuda_prefix_sum/subtile_kernel_launcher.cuh"

__global__ void SubtileKernel(
    // int *d_data,
    KernelLaunchParams params
) {
  // Declare dynamic shared memory
  extern __shared__ int shared_mem[];

  // Divide shared memory into two arrays
  KernelArray array_a{.d_address = shared_mem, .size = params.array.size};

  // === Phase 1: Load input from global memory to shared memory ===
  // CopyGlobalArrayToSharedArray(params.array, array_a, params.tile_size);
  CopyMETTiledArray(params.array, array_a, params.tile_size);
  __syncthreads();

  // === Phase 2: Row-wise prefix sum within each tile of arrayA ===
  ComputeLocalRowWisePrefixSums(array_a, params.tile_size);
  __syncthreads();

  // === Phase 3: Column-wise prefix sum within each tile of arrayA ===
  ComputeLocalColWisePrefixSums(array_a, params.tile_size);
  __syncthreads();

  // === Phase 4: Broadcast right edge values to downstream elements ===
  BroadcastRightEdgesInPlace(array_a, params.tile_size);
  __syncthreads();

  // === Phase 5: Broadcast bottom edge values to downstream elements ===
  BroadcastBottomEdgesInPlace(array_a, params.tile_size);
  __syncthreads();

  // === Phase 6: Write final result back to global memory ===
  CopyMETTiledArray(array_a, params.array, params.tile_size);
}


void LaunchSubtileKernel(KernelLaunchParams kernel_params) {

  int num_tile_cols =
      kernel_params.array.size.num_cols / kernel_params.tile_size.num_cols;
  int num_tile_rows =
      kernel_params.array.size.num_rows / kernel_params.tile_size.num_rows;

  dim3 blockDim(num_tile_cols, num_tile_rows);
  dim3 gridDim(1, 1); // Single block for now

  int shared_mem_size = kernel_params.array.size.num_rows *
                        kernel_params.array.size.num_cols * sizeof(int);

  ConfigureSharedMemoryForKernel(SubtileKernel, 98304);

  SubtileKernel<<<gridDim, blockDim, shared_mem_size, 0>>>(kernel_params);

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
