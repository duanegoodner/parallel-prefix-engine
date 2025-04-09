#pragma once


#include "cuda_prefix_sum/kernel_launch_params.hpp"


__device__ void PrintSharedMemoryArrayNew(
    const int *array,
    int full_matrix_dim_x,
    int full_matrix_dim_y,
    const char *label
) {
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    printf("%s:\n", label);
    for (int row = 0; row < full_matrix_dim_x; ++row) {
      for (int col = 0; col < full_matrix_dim_y; ++col) {
        printf("%d\t", array[row * full_matrix_dim_y + col]);
      }
      printf("\n");
    }
  }
}

__device__ int GetIndex1D(int x_idx, int y_idx, int array_size_y) {
    return x_idx * array_size_y + y_idx;
}

__device__ int GetFullMatrixIndexX(
    int tile_row,
    int tile_col,
    int tile_dim_x
) {
  return threadIdx.x * tile_dim_x + tile_row;
}

__device__ int GetFullMatrixIndexY(
    int tile_row,
    int tile_col,
    int tile_dim_y
) {
  return threadIdx.y * tile_dim_y + tile_col;
}