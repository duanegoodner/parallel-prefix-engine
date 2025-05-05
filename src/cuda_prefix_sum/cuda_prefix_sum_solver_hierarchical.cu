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
  const int array_width = params.array.size.y;
  const int array_height = params.array.size.x;
  const int tile_width = params.tile_size.y;
  const int tile_height = params.tile_size.x;

  extern __shared__ int tile_smem[];

  int tid = threadIdx.x;
  int local_x = tid / tile_width;
  int local_y = tid % tile_width;

  if (local_x >= tile_height || local_y >= tile_width)
    return;

  int tile_i = blockIdx.x;
  int tile_j = blockIdx.y;

  int global_x = tile_i * tile_height + local_x;
  int global_y = tile_j * tile_width + local_y;

  if (global_x >= array_height || global_y >= array_width)
    return;

  int global_idx = global_x * array_width + global_y;

  // === Phase 1: Load into shared memory ===
  tile_smem[index_2d(local_x, local_y, tile_width)] =
      params.array.d_address[global_idx];
  __syncthreads();

  // === Phase 2: Local 2D prefix sum ===
  ComputeTilePrefixSumInShared(
      tile_smem,
      tile_height,
      tile_width,
      local_x,
      local_y
  );
  __syncthreads();

  // === Phase 3: Write result back ===
  params.array.d_address[global_idx] =
      tile_smem[index_2d(local_x, local_y, tile_width)];
  __syncthreads();

  // === Phase 4: Collect right edge ===
  if (tile_j < gridDim.y - 1 && local_y == tile_width - 1) {
    int edge_idx = global_x * (gridDim.y - 1) + tile_j;
    right_edges[edge_idx] =
        tile_smem[index_2d(local_x, tile_width - 1, tile_width)];
  }

  // === Phase 5: Collect bottom edge ===
  if (tile_i < gridDim.x - 1 && local_x == tile_height - 1) {
    int edge_idx = tile_i * array_width + global_y;
    bottom_edges[edge_idx] =
        tile_smem[index_2d(tile_height - 1, local_y, tile_width)];
  }
}

void LaunchPrefixSumKernelHierarchical(KernelLaunchParams params) {
  const int array_height = params.array.size.x;
  const int array_width = params.array.size.y;
  const int tile_height = params.tile_size.x;
  const int tile_width = params.tile_size.y;

  // Compute tile grid
  int tiles_x = (array_height + tile_height - 1) / tile_height;
  int tiles_y = (array_width + tile_width - 1) / tile_width;

  // Create and allocate workspace internally
  PrefixSumTileWorkspace workspace;
  workspace.Allocate(array_height, array_width, tile_height, tile_width);

  // Kernel launch config
  dim3 grid(tiles_x, tiles_y);
  dim3 block(tile_height * tile_width);
  size_t shared_mem_bytes = tile_height * tile_width * sizeof(int);

  // Launch the hierarchical prefix sum kernel
  PrefixSumKernelHierarchical<<<grid, block, shared_mem_bytes>>>(
      params,
      workspace.d_right_edges,
      workspace.d_bottom_edges
  );

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }

  cudaDeviceSynchronize();

  // Clean up workspace
  workspace.Free();
}