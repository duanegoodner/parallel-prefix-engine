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

__device__ int ArrayIndexX(int tile_row, int tile_size_x) {
  return threadIdx.x * tile_size_x + tile_row;
}

__device__ int ArrayIndexY(int tile_col, int tile_size_y) {
  return threadIdx.y * tile_size_y + tile_col;
}

__device__ void PrintTileContents(
    KernelArray array,
    ArraySize2D tile_size,
    int tile_row = 0,
    int tile_col = 0
) {
  if (threadIdx.x == tile_row && threadIdx.y == tile_col) {

    PrintThreadAndBlockIndices();
    for (int tile_x = 0; tile_x < tile_size.x; ++tile_x) {
      for (int tile_y = 0; tile_y < tile_size.y; ++tile_y) {
        int arr_idx_x = ArrayIndexX(tile_x, tile_size.x);
        int arr_idx_y = ArrayIndexY(tile_y, tile_size.y);
        int index_1d = ArrayIndex1D(arr_idx_x, arr_idx_y, array.size.y);
        printf("%d\t", array.d_address[index_1d]);
      }
      printf("\n");
    }
  }
}

__device__ void CopyGlobalArrayToSharedArray(
    KernelArray global_array,
    KernelArray shared_array,
    ArraySize2D tile_size
) {
  for (int tile_row = 0; tile_row < tile_size.x; ++tile_row) {
    for (int tile_col = 0; tile_col < tile_size.y; ++tile_col) {
      int arr_idx_x = ArrayIndexX(tile_row, tile_size.x);
      int arr_idx_y = ArrayIndexY(tile_col, tile_size.y);
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
      int arr_idx_x = ArrayIndexX(tile_row, tile_size.x);
      int arr_idx_y = ArrayIndexY(tile_col, tile_size.y);
      int index_1d = ArrayIndex1D(arr_idx_x, arr_idx_y, global_array.size.y);
      global_array.d_address[index_1d] = shared_array.d_address[index_1d];
    }
  }
}

__device__ void CopySharedArrayToSharedArray(
    KernelArray source_array,
    KernelArray dest_array,
    ArraySize2D tile_size
) {

  for (int tile_row = 0; tile_row < tile_size.x; ++tile_row) {
    for (int tile_col = 0; tile_col < tile_size.y; ++tile_col) {
      int arr_idx_x = ArrayIndexX(tile_row, tile_size.x);
      int arr_idx_y = ArrayIndexY(tile_col, tile_size.y);
      int index_1d = ArrayIndex1D(arr_idx_x, arr_idx_y, source_array.size.y);

      dest_array.d_address[index_1d] = source_array.d_address[index_1d];
    }
  }
}

__device__ void LoadFromGlobalToSharedMemory(
    int *global_array,
    int *local_array,
    ArraySize2D array_size,
    ArraySize2D tile_size
) {
  for (int tile_row = 0; tile_row < tile_size.x; ++tile_row) {
    for (int tile_col = 0; tile_col < tile_size.y; ++tile_col) {
      int arr_idx_x = ArrayIndexX(tile_row, tile_size.x);
      int arr_idx_y = ArrayIndexY(tile_col, tile_size.y);
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
    printf("\n");
  }
}

__device__ void PrintKernelArrayFlattended(
    KernelArray array,
    const char *label
) {
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    printf("%s:\n", label);
    for (int index_1d = 0; index_1d < array.size.x * array.size.y;
         ++index_1d) {
      printf("%d\t", array.d_address[index_1d]);
    }
    printf("\n");
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
    KernelArray arr,
    ArraySize2D tile_size,
    int tile_row,
    int tile_col
) {
  int index_x = ArrayIndexX(tile_row, tile_size.x);
  int index_y = ArrayIndexY(tile_col, tile_size.y);
  int index_y_prev = ArrayIndexY(tile_col - 1, tile_size.y);
  CombineElementInto(arr, index_x, index_y_prev, index_x, index_y);
}

__device__ void ComputeColWisePrefixSum(
    KernelArray arr,
    ArraySize2D tile_size,
    int tile_row,
    int tile_col
) {
  int index_x = ArrayIndexX(tile_row, tile_size.x);
  int index_y = ArrayIndexY(tile_col, tile_size.y);
  int index_x_prev = ArrayIndexX(tile_row - 1, tile_size.x);
  CombineElementInto(arr, index_x_prev, index_y, index_x, index_y);
}

__device__ void SumAndCopy(
    KernelArray source_array,
    int cur_x,
    int cur_y,
    int val_to_add,
    KernelArray dest_array
) {
  int index_1d = ArrayIndex1D(cur_x, cur_y, source_array.size.y);
  dest_array.d_address[index_1d] =
      source_array.d_address[index_1d] + val_to_add;
}

struct AccumulateRight {
  __device__ void operator()(
      KernelArray src,
      ArraySize2D tile_size,
      int x,
      int y,
      int &acc
  ) const {
    for (int block_col = 0; block_col < threadIdx.y; ++block_col) {
      int upstream_y = block_col + tile_size.y - 1;
      int idx = ArrayIndex1D(x, upstream_y, src.size.y);
      acc += src.d_address[idx];
    }
  }
};

struct AccumulateBottom {
  __device__ void operator()(
      KernelArray src,
      ArraySize2D tile_size,
      int x,
      int y,
      int &acc
  ) const {
    for (int block_row = 0; block_row < threadIdx.x; ++block_row) {
      int upstream_x = block_row + tile_size.x - 1;
      int idx = ArrayIndex1D(upstream_x, y, src.size.y);
      acc += src.d_address[idx];
    }
  }
};

template <typename AccumulateFn>
__device__ void AccumulateEdges(
    KernelArray src,
    ArraySize2D tile_size,
    KernelArray dst,
    AccumulateFn accumulate_fn
) {
  for (int tile_row = 0; tile_row < tile_size.x; ++tile_row) {
    for (int tile_col = 0; tile_col < tile_size.y; ++tile_col) {
      int x = ArrayIndexX(tile_row, tile_size.x);
      int y = ArrayIndexY(tile_col, tile_size.y);
      int idx = ArrayIndex1D(x, y, src.size.y);

      int sum = src.d_address[idx];

      // Functor decides what to do with x, y, tile
      accumulate_fn(src, tile_size, x, y, sum);

      dst.d_address[idx] = sum;
    }
  }
}

__device__ void BroadcastRightEdges(
    KernelArray source_array,
    ArraySize2D tile_size,
    KernelArray dest_array
) {
  // iterate over each tile to right of cur_tile
  for (int block_col = threadIdx.y + 1; block_col < blockDim.y; ++block_col) {
    // iterate over each row of cur_tile
    for (int tile_row = 0; tile_row < tile_size.x; ++tile_row) {
      // iterate over each column of downstream tile
      for (int downstream_tile_col = 0; downstream_tile_col < tile_size.y;
           ++downstream_tile_col) {
        int array_x = threadIdx.x * tile_size.x + tile_row;
        int source_array_y = ArrayIndexY(tile_size.y - 1, tile_size.y);
        int dest_array_y = block_col * tile_size.y + downstream_tile_col;
        int dest_index_1d =
            ArrayIndex1D(array_x, dest_array_y, dest_array.size.y);
        int source_index_1d =
            ArrayIndex1D(array_x, source_array_y, source_array.size.y);
        dest_array.d_address[dest_index_1d] +=
            source_array.d_address[source_index_1d];
      }
    }
  }
}

__device__ void BroadcastBottomEdges(
    KernelArray source_array,
    ArraySize2D tile_size,
    KernelArray dest_array
) {

  // iterate over each tile below cur_tile
  for (int block_row = threadIdx.x + 1; block_row < blockDim.x; ++block_row) {
    // iterate over each col of cur_tile
    for (int tile_col = 0; tile_col < tile_size.y; ++tile_col) {
      // iterate over each row of downstream tile
      for (int downstream_tile_row = 0; downstream_tile_row < tile_size.x;
           ++downstream_tile_row) {

        int array_y = threadIdx.y * tile_size.y + tile_col;
        int source_array_x = ArrayIndexX(tile_size.x - 1, tile_size.x);
        int dest_array_x = block_row * tile_size.x + downstream_tile_row;

        int dest_index_1d =
            ArrayIndex1D(dest_array_x, array_y, dest_array.size.y);
        int source_index_1d =
            ArrayIndex1D(source_array_x, array_y, source_array.size.y);
        dest_array.d_address[dest_index_1d] +=
            source_array.d_address[source_index_1d];
      }
    }
  }
}



// hierarchichal helpers

