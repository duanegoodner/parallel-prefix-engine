#pragma once

#include "cuda_prefix_sum/kernel_launch_params.hpp"

__device__ int Index1D(int x_idx, int y_idx, int array_size_y) {
  return x_idx * array_size_y + y_idx;
}

__device__ int FullArrayX(int tile_row, int tile_col, int tile_size_x) {
  return threadIdx.x * tile_size_x + tile_row;
}

__device__ int FullArrayY(int tile_row, int tile_col, int tile_size_y) {
  return threadIdx.y * tile_size_y + tile_col;
}

__device__ void LoadFromGlobalToSharedMemory(
    int *d_data,
    int *local_array,
    KernelLaunchParams params
) {
  // Load data from global memory to shared memory
  for (int tile_row = 0; tile_row < params.tile_size_x; ++tile_row) {
    for (int tile_col = 0; tile_col < params.tile_size_y; ++tile_col) {
      int full_matrix_x = threadIdx.x * params.tile_size_x + tile_row;
      int full_matrix_y = threadIdx.y * params.tile_size_y + tile_col;
      local_array[full_matrix_x * params.full_matrix_dim_y + full_matrix_y] =
          d_data[full_matrix_x * params.full_matrix_dim_y + full_matrix_y];
    }
  }
}

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

__device__ void CombineElementInto(
    int *local_array,
    int full_matrix_dim_x,
    int full_matrix_dim_y,
    int other_element_full_matrix_x,
    int other_element_full_matrix_y,
    int cur_element_full_matrix_x,
    int cur_element_full_matrix_y
) {

  local_array
      [cur_element_full_matrix_x * full_matrix_dim_y +
       cur_element_full_matrix_y] += local_array
          [other_element_full_matrix_x * full_matrix_dim_y +
           other_element_full_matrix_y];
}

__device__ void ComputeRowWisePrefixSum(
    int *full_matrix,
    int full_matrix_dim_x,
    int full_matrix_dim_y,
    int tile_row,
    int tile_col,
    int tile_size_x,
    int tile_size_y
) {
  int full_matrix_x = FullArrayX(tile_row, tile_col, tile_size_x);
  int full_matrix_y = FullArrayY(tile_row, tile_col, tile_size_y);
  int full_matrix_y_prev = FullArrayY(tile_row, tile_col - 1, tile_size_y);
  CombineElementInto(
      full_matrix,
      full_matrix_dim_x,
      full_matrix_dim_y,
      full_matrix_x,
      full_matrix_y_prev,
      full_matrix_x,
      full_matrix_y
  );
}