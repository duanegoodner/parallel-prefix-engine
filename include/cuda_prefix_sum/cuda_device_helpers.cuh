#pragma once

#include "cuda_prefix_sum/kernel_launch_params.hpp"

// Debug statement: Print thread and block indices
__device__ void PrintThreadAndBlockIndices() {
  printf(
      "Block (%d, %d), Thread (%d, %d), Global Index: %d\n",
      blockIdx.x,
      blockIdx.y,
      threadIdx.x,
      threadIdx.y,
      threadIdx.x * blockDim.y + threadIdx.y
  );
}

__device__ int ArrayIndex1D(int x_idx, int y_idx, int array_size_y) {
  return x_idx * array_size_y + y_idx;
}

__device__ int ArrayIndexX(int tile_row, int tile_col, int tile_size_x) {
  return threadIdx.x * tile_size_x + tile_row;
}

__device__ int ArrayIndexY(int tile_row, int tile_col, int tile_size_y) {
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
      int arr_idx_x = ArrayIndexX(tile_row, tile_col, params.tile_size_x);
      int arr_idx_y = ArrayIndexY(tile_row, tile_col, params.tile_size_y);
      int index_1d = ArrayIndex1D(arr_idx_x, arr_idx_y, params.arr_size_y);
      local_array[index_1d] = d_data[index_1d];
    }
  }
}

__device__ void PrintArray(
    const int *arr,
    int arr_size_x,
    int arr_size_y,
    const char *label
) {
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    printf("%s:\n", label);
    for (int row = 0; row < arr_size_x; ++row) {
      for (int col = 0; col < arr_size_y; ++col) {
        printf("%d\t", arr[ArrayIndex1D(row, col, arr_size_y)]);
      }
      printf("\n");
    }
  }
}

__device__ void CombineElementInto(
    int *arr,
    int arr_size_x,
    int arr_size_y,
    int other_index_x,
    int other_index_y,
    int cur_index_x,
    int cur_index_y
) {

  int cur_idx_1d = ArrayIndex1D(cur_index_x, cur_index_y, arr_size_y);
  int other_idx_1d =
      ArrayIndex1D(other_index_x, other_index_y, arr_size_y);
  arr[cur_idx_1d] += arr[other_idx_1d];
}

__device__ void ComputeRowWisePrefixSum(
    int *arr,
    int arr_size_x,
    int arr_size_y,
    int tile_row,
    int tile_col,
    int tile_size_x,
    int tile_size_y
) {
  int index_x = ArrayIndexX(tile_row, tile_col, tile_size_x);
  int index_y = ArrayIndexY(tile_row, tile_col, tile_size_y);
  int index_y_prev = ArrayIndexY(tile_row, tile_col - 1, tile_size_y);
  CombineElementInto(
      arr,
      arr_size_x,
      arr_size_y,
      index_x,
      index_y_prev,
      index_x,
      index_y
  );
}