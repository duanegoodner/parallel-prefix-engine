#pragma once

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