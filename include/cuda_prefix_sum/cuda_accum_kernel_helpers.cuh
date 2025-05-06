#pragma once

#include "cuda_prefix_sum/kernel_launch_params.hpp"



__device__ inline int index_2d(int x, int y, int width) {
  return x * width + y;
}

__device__ void ComputeTilePrefixSumInShared(
    int *tile_smem,
    int tile_height,
    int tile_width,
    int local_x,
    int local_y
) {
  // Row-wise inclusive scan
  int val = tile_smem[index_2d(local_x, local_y, tile_width)];
  for (int i = 1; i <= local_y; ++i) {
    val += tile_smem[index_2d(local_x, local_y - i, tile_width)];
  }
  __syncthreads();
  tile_smem[index_2d(local_x, local_y, tile_width)] = val;
  __syncthreads();

  // Column-wise inclusive scan
  val = tile_smem[index_2d(local_x, local_y, tile_width)];
  for (int i = 1; i <= local_x; ++i) {
    val += tile_smem[index_2d(local_x - i, local_y, tile_width)];
  }
  __syncthreads();
  tile_smem[index_2d(local_x, local_y, tile_width)] = val;
  __syncthreads();
}

__device__ void ComputeAndStoreTilePrefixSum(
    KernelLaunchParams params,
    int *tile_smem,
    int tile_height,
    int tile_width,
    int local_x,
    int local_y,
    int global_x,
    int global_y,
    int tile_i,
    int tile_j,
    int *right_edges,  // [array_height][tiles_y - 1]
    int *bottom_edges, // [tiles_x - 1][array_width]
    int grid_dim_x,
    int grid_dim_y
) {
  const int array_width = params.array.size.num_cols;
  const int array_height = params.array.size.num_rows;

  // Check bounds
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
  if (tile_j < grid_dim_y - 1 && local_y == tile_width - 1) {
    int edge_idx = global_x * (grid_dim_y - 1) + tile_j;
    right_edges[edge_idx] =
        tile_smem[index_2d(local_x, tile_width - 1, tile_width)];
  }

  // === Phase 5: Collect bottom edge ===
  if (tile_i < grid_dim_x - 1 && local_x == tile_height - 1) {
    int edge_idx = tile_i * array_width + global_y;
    bottom_edges[edge_idx] =
        tile_smem[index_2d(tile_height - 1, local_y, tile_width)];
  }
}



// Hierarchical kernal helpers

