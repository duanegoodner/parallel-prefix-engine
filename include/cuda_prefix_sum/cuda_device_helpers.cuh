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

__device__ void CopyGlobalArrayToSharedArray(
    KernelArray global_array,
    KernelArray shared_array,
    ArraySize2D tile_size
) {
  for (int tile_row = 0; tile_row < tile_size.x; ++tile_row) {
    for (int tile_col = 0; tile_col < tile_size.y; ++tile_col) {
      int arr_idx_x = ArrayIndexX(tile_row, tile_col, tile_size.x);
      int arr_idx_y = ArrayIndexY(tile_row, tile_col, tile_size.y);
      int index_1d = ArrayIndex1D(arr_idx_x, arr_idx_y, global_array.size.y);
      shared_array.d_address[index_1d] = global_array.d_address[index_1d];
    }
  }
}

__device__ void CopySharedArrayToGlobalArray(
    KernelArray shared_array,
    KernelArray global_array,
    ArraySize2D tile_size
) {
  for (int tile_row = 0; tile_row < tile_size.x; ++tile_row) {
    for (int tile_col = 0; tile_col < tile_size.y; ++tile_col) {
      int arr_idx_x = ArrayIndexX(tile_row, tile_col, tile_size.x);
      int arr_idx_y = ArrayIndexY(tile_row, tile_col, tile_size.y);
      int index_1d = ArrayIndex1D(arr_idx_x, arr_idx_y, global_array.size.y);
      global_array.d_address[index_1d] = shared_array.d_address[index_1d];
    }
  }
}

__device__ void LoadFromGlobalToSharedMemory(
    int *global_array,
    int *local_array,
    ArraySize2D array_size,
    ArraySize2D tile_size
    // KernelLaunchParams params
) {
  // Load data from global memory to shared memory
  for (int tile_row = 0; tile_row < tile_size.x; ++tile_row) {
    for (int tile_col = 0; tile_col < tile_size.y; ++tile_col) {
      int arr_idx_x = ArrayIndexX(tile_row, tile_col, tile_size.x);
      int arr_idx_y = ArrayIndexY(tile_row, tile_col, tile_size.y);
      int index_1d = ArrayIndex1D(arr_idx_x, arr_idx_y, array_size.y);
      local_array[index_1d] = global_array[index_1d];
    }
  }
}

__device__ void PrintKernelArray(KernelArray array, const char *label) {
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    printf("%s:\n", label);
    for (int row = 0; row < array.size.x; ++row) {
      for (int col = 0; col < array.size.y; ++col) {
        printf("%d\t", array.d_address[ArrayIndex1D(row, col, array.size.y)]);
      }
      printf("\n");
    }
  }
}

__device__ void
PrintArray(const int *arr, int arr_size_x, int arr_size_y, const char *label) {
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
    KernelArray arr,
    int other_index_x,
    int other_index_y,
    int cur_index_x,
    int cur_index_y
) {

  int cur_idx_1d = ArrayIndex1D(cur_index_x, cur_index_y, arr.size.y);
  int other_idx_1d = ArrayIndex1D(other_index_x, other_index_y, arr.size.y);
  arr.d_address[cur_idx_1d] += arr.d_address[other_idx_1d];
}

__device__ void ComputeRowWisePrefixSum(
    // int *arr,
    // int arr_size_x,
    // int arr_size_y,
    KernelArray arr,
    ArraySize2D tile_size,
    int tile_row,
    int tile_col
    // int tile_size_x,
    // int tile_size_y
) {
  int index_x = ArrayIndexX(tile_row, tile_col, tile_size.x);
  int index_y = ArrayIndexY(tile_row, tile_col, tile_size.y);
  int index_y_prev = ArrayIndexY(tile_row, tile_col - 1, tile_size.y);
  CombineElementInto(
      arr,
      // arr_size_x,
      // arr_size_y,
      index_x,
      index_y_prev,
      index_x,
      index_y
  );
}