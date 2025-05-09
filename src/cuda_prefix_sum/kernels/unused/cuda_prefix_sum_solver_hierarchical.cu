#include <cuda_runtime.h>

#include <cstdio>
#include <iostream>

#include "cuda_prefix_sum/cuda_accum_kernel_helpers.cuh"
#include "cuda_prefix_sum/cuda_prefix_sum_solver.cuh"
#include "cuda_prefix_sum/kernel_launch_params.hpp"
#include "cuda_prefix_sum/prefix_sum_tile_workspace.hpp"

__global__ void PrefixSumKernelHierarchical(
    KernelLaunchParams params,
    int *right_edges, // [array_height][tiles_y - 1]
    int *bottom_edges // [tiles_x - 1][array_width]
) {
  const int array_width = params.array.size.num_cols;
  const int array_height = params.array.size.num_rows;
  const int tile_width = params.tile_size.num_cols;
  const int tile_height = params.tile_size.num_rows;

  int tiles_x = gridDim.x;
  int tiles_y = gridDim.y;

  int tile_i = blockIdx.x;
  int tile_j = blockIdx.y;

  int tid = threadIdx.x;
  int local_x = tid / tile_width;
  int local_y = tid % tile_width;

  if (local_x >= tile_height || local_y >= tile_width)
    return;

  int global_x = tile_i * tile_height + local_x;
  int global_y = tile_j * tile_width + local_y;

  if (global_x >= array_height || global_y >= array_width)
    return;

  extern __shared__ int tile_smem[];

  // === Step 1: Local tile scan + collect edges ===
  ComputeAndStoreTilePrefixSum(
      params,
      tile_smem,
      tile_height,
      tile_width,
      local_x,
      local_y,
      global_x,
      global_y,
      tile_i,
      tile_j,
      right_edges,
      bottom_edges,
      tiles_x,
      tiles_y
  );

  __syncthreads();

  // === Step 2: Compute inter-tile offset ===
  int offset = 0;

  // Row offset
  for (int tj = 0; tj < tile_j; ++tj) {
    if (global_x < array_height) {
      int edge_idx = global_x * (tiles_y - 1) + tj;
      offset += right_edges[edge_idx];
    }
  }

  // Column offset
  for (int ti = 0; ti < tile_i; ++ti) {
    if (global_y < array_width) {
      int edge_idx = ti * array_width + global_y;
      offset += bottom_edges[edge_idx];
    }
  }

  __syncthreads();

  // === Step 3: Apply offset
  int global_idx = global_x * array_width + global_y;
  params.array.d_address[global_idx] += offset;
}

void LaunchPrefixSumKernelHierarchical(KernelLaunchParams params) {
  int array_height = params.array.size.num_rows;
  int array_width = params.array.size.num_cols;
  int tile_height = params.tile_size.num_rows;
  int tile_width = params.tile_size.num_cols;

  int tiles_x = (array_height + tile_height - 1) / tile_height;
  int tiles_y = (array_width + tile_width - 1) / tile_width;

  PrefixSumTileWorkspace workspace;
  workspace.Allocate(array_height, array_width, tile_height, tile_width);

  dim3 grid(tiles_x, tiles_y);
  dim3 block(tile_height * tile_width);
  size_t shared_mem_bytes = tile_height * tile_width * sizeof(int);

  PrefixSumKernelHierarchical<<<grid, block, shared_mem_bytes>>>(
      params,
      workspace.d_right_edges,
      workspace.d_bottom_edges
  );

  cudaDeviceSynchronize();
  workspace.Free();
}
