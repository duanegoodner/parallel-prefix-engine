#include <cuda_runtime.h>

#include <cstdio>
#include <iostream>

// #include "cuda_prefix_sum/cuda_met_device_helpers.cuh"
// #include "cuda_prefix_sum/cuda_accum_kernel_helpers.cuh"
// #include "cuda_prefix_sum/cuda_prefix_sum_solver.cuh"
#include "cuda_prefix_sum/kernel_launch_params.hpp"
// #include "cuda_prefix_sum/prefix_sum_tile_workspace.hpp"

__device__ inline int index_2d(int x, int y, int width) {
  return x * width + y;
}

__global__ void PrefixSumKernelAccum(KernelLaunchParams params) {
  const int array_width = params.array.size.num_rows;
  const int array_height = params.array.size.num_cols;
  const int tile_width = params.tile_size.num_rows;
  const int tile_height = params.tile_size.num_cols;

  extern __shared__ int tile[]; // size = tile_width * tile_height

  int tid = threadIdx.x;
  int local_x = tid / tile_width; // tile row
  int local_y = tid % tile_width; // tile column

  if (local_x >= tile_height || local_y >= tile_width)
    return;

  int global_x = blockIdx.x * tile_height + local_x;
  int global_y = blockIdx.y * tile_width + local_y;

  if (global_x >= array_height || global_y >= array_width)
    return;

  int global_idx = index_2d(global_x, global_y, array_width);

  // === Phase 1: Load into shared memory ===
  tile[index_2d(local_x, local_y, tile_width)] =
      params.array.d_address[global_idx];
  __syncthreads();

  // === Phase 2: Row-wise inclusive scan (per row) ===
  int val = tile[index_2d(local_x, local_y, tile_width)];
  for (int i = 1; i <= local_y; ++i) {
    val += tile[index_2d(local_x, local_y - i, tile_width)];
  }
  __syncthreads();
  tile[index_2d(local_x, local_y, tile_width)] = val;
  __syncthreads();

  // === Phase 3: Column-wise inclusive scan (per column) ===
  val = tile[index_2d(local_x, local_y, tile_width)];
  for (int i = 1; i <= local_x; ++i) {
    val += tile[index_2d(local_x - i, local_y, tile_width)];
  }
  __syncthreads();
  tile[index_2d(local_x, local_y, tile_width)] = val;
  __syncthreads();

  // === Phase 4: Write result back in-place ===
  params.array.d_address[global_idx] =
      tile[index_2d(local_x, local_y, tile_width)];
}

void LaunchPrefixSumKernelAccum(KernelLaunchParams kernel_params) {
  int array_width = kernel_params.array.size.num_rows;
  int array_height = kernel_params.array.size.num_cols;
  int tile_width = kernel_params.tile_size.num_rows;
  int tile_height = kernel_params.tile_size.num_cols;

  int threads_per_tile = tile_width * tile_height;
  size_t shared_mem_bytes = threads_per_tile * sizeof(int);

  dim3 grid(
      (array_width + tile_width - 1) /
          tile_width, // grid.x → blocks across width
      (array_height + tile_height - 1) /
          tile_height // grid.y → blocks down height
  );

  dim3 block(threads_per_tile);

  PrefixSumKernelAccum<<<grid, block, shared_mem_bytes>>>(kernel_params);

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
