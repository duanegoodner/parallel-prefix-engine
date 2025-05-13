#pragma once

#include "cuda_prefix_sum/internal/kernel_launch_params.hpp"

#pragma once

#include "cuda_prefix_sum/internal/kernel_launch_params.hpp"

//
// ElementCoords: Represents the position of a single element
// within a thread's subtile, with helpers to compute its location
// in global memory and shared memory.
//

#pragma once

#include "cuda_prefix_sum/internal/kernel_launch_params.hpp"

//
// Row-major 1D index calculation
//
__forceinline__ __device__ int ArrayIndex1D(int row, int col, int num_cols) {
  return row * num_cols + col;
}

//
// ElementCoords:
// Encapsulates per-element indexing logic for a thread's subtile,
// including automatic global and shared memory access.
//
struct ElementCoords {
  int local_row;
  int local_col;
  ArraySize2D subtile_size;

  __device__ int global_row() const {
    return blockIdx.y * blockDim.y * subtile_size.num_rows +
           threadIdx.y * subtile_size.num_rows + local_row;
  }

  __device__ int global_col() const {
    return blockIdx.x * blockDim.x * subtile_size.num_cols +
           threadIdx.x * subtile_size.num_cols + local_col;
  }

  __device__ int shared_row() const {
    return threadIdx.y * subtile_size.num_rows + local_row;
  }

  __device__ int shared_col() const {
    return threadIdx.x * subtile_size.num_cols + local_col;
  }

  __device__ int GlobalArrayIndex1D(int global_width) const {
    return ArrayIndex1D(global_row(), global_col(), global_width);
  }

  __device__ int SharedArrayIndex1D() const {
    int tile_width = blockDim.x * subtile_size.num_cols;
    return ArrayIndex1D(shared_row(), shared_col(), tile_width);
  }
};

//
// Helper to construct an ElementCoords instance
//
__forceinline__ __device__ ElementCoords GetElementCoords(
    int local_row,
    int local_col,
    const ArraySize2D &subtile_size
) {
  return ElementCoords{local_row, local_col, subtile_size};
}

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

__device__ int GetSharedMemCol(int local_col, int tile_size_x) {
  return threadIdx.x * tile_size_x + local_col;
}

__device__ int GetSharedMemRow(int local_row, int tile_size_y) {
  return threadIdx.y * tile_size_y + local_row;
}

__device__ void PrintSubTileContents(
    KernelArray shared_mem_array,
    ArraySize2D sub_tile_size,
    int tile_row_index = 0,
    int tile_col_index = 0,
    int sub_tile_row_index = 0,
    int sub_tile_col_index = 0
) {
  if (blockIdx.x == tile_col_index && blockIdx.y == tile_row_index &&
      threadIdx.x == sub_tile_col_index && threadIdx.y == sub_tile_row_index) {

    PrintThreadAndBlockIndices();
    for (int local_row = 0; local_row < sub_tile_size.num_rows; ++local_row) {
      for (int local_col = 0; local_col < sub_tile_size.num_cols;
           ++local_col) {
        auto coords = GetElementCoords(local_row, local_col, sub_tile_size);
        auto shared_mem_index_1d = coords.SharedArrayIndex1D();
        printf(" %d", shared_mem_array.d_address[shared_mem_index_1d]);
      }
      printf("\n");
    }
  }
}

__device__ void CopyFromGlobalToShared(
    KernelArray global_array,
    KernelArray shared_mem_array,
    ArraySize2D sub_tile_size
) {

  for (int local_row = 0; local_row < sub_tile_size.num_rows; ++local_row) {
    for (int local_col = 0; local_col < sub_tile_size.num_cols; ++local_col) {
      auto coords = GetElementCoords(local_row, local_col, sub_tile_size);
      auto global_index_1d =
          coords.GlobalArrayIndex1D(global_array.size.num_cols);
      auto shared_mem_index_1d = coords.SharedArrayIndex1D();
      shared_mem_array.d_address[shared_mem_index_1d] =
          global_array.d_address[global_index_1d];
    }
  }
}

__device__ void CopyFromSharedToGlobal(
    KernelArray shared_mem_array,
    KernelArray global_array,
    ArraySize2D sub_tile_size
) {

  for (int local_row = 0; local_row < sub_tile_size.num_rows; ++local_row) {
    for (int local_col = 0; local_col < sub_tile_size.num_cols; ++local_col) {
      auto coords = GetElementCoords(local_row, local_col, sub_tile_size);
      auto global_index_1d =
          coords.GlobalArrayIndex1D(global_array.size.num_cols);
      auto shared_mem_index_1d = coords.SharedArrayIndex1D();
      global_array.d_address[shared_mem_index_1d] =
          shared_mem_array.d_address[global_index_1d];
    }
  }
}

__device__ void PrintKernelArray(KernelArray array, const char *label) {
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    printf("%s:\n", label);
    for (int row = 0; row < array.size.num_rows; ++row) {
      for (int col = 0; col < array.size.num_cols; ++col) {
        printf(
            "%d\t",
            array.d_address[ArrayIndex1D(row, col, array.size.num_cols)]
        );
      }
      printf("\n");
    }
    printf("\n");
  }
}

__device__ void CombineElementInto(
    KernelArray shared_mem_array,
    ElementCoords other_element,
    ElementCoords cur_element
) {
  shared_mem_array.d_address[cur_element.SharedArrayIndex1D()] +=
      shared_mem_array.d_address[other_element.SharedArrayIndex1D()];
}

__device__ void ComputeLocalRowWisePrefixSums(
    KernelArray shared_mem_array,
    ArraySize2D sub_tile_size
) {
  __syncthreads();
  for (int local_col = 1; local_col < sub_tile_size.num_cols; ++local_col) {
    for (int local_row = 0; local_row < sub_tile_size.num_rows; ++local_row) {
      auto cur_coords = GetElementCoords(local_row, local_col, sub_tile_size);
      auto other_coords =
          GetElementCoords(local_row, local_col - 1, sub_tile_size);
      CombineElementInto(shared_mem_array, other_coords, cur_coords);
    }
  }
}

__device__ void ComputeLocalColWisePrefixSums(
    KernelArray shared_mem_array,
    ArraySize2D sub_tile_size
) {

  for (int local_row = 1; local_row < sub_tile_size.num_rows; ++local_row) {
    for (int local_col = 0; local_col < sub_tile_size.num_cols; ++local_col) {
      auto cur_coords = GetElementCoords(local_row, local_col, sub_tile_size);
      auto other_coords =
          GetElementCoords(local_row - 1, local_col, sub_tile_size);
      CombineElementInto(shared_mem_array, other_coords, cur_coords);
    }
  }
}

__device__ void BroadcastRightEdgesInPlace(
    KernelArray arr,
    ArraySize2D tile_size
) {
  // Each thread is responsible for broadcasting from one row
  for (int local_row = 0; local_row < tile_size.num_rows; ++local_row) {
    int full_row = GetSharedMemRow(local_row, tile_size.num_rows);

    // === Step 1: Preload the source value *once*
    int source_col =
        GetSharedMemCol(tile_size.num_cols - 1, tile_size.num_cols);
    int idx_src = ArrayIndex1D(full_row, source_col, arr.size.num_cols);
    int edge_val = arr.d_address[idx_src];

    __syncthreads(); // Ensure all source values are read before any writes

    // === Step 2: Apply edge_val to downstream tiles
    for (int block_col = threadIdx.x + 1; block_col < blockDim.x;
         ++block_col) {
      for (int downstream_col = 0; downstream_col < tile_size.num_cols;
           ++downstream_col) {
        int target_col = block_col * tile_size.num_cols + downstream_col;
        int idx_dst = ArrayIndex1D(full_row, target_col, arr.size.num_cols);
        arr.d_address[idx_dst] += edge_val;
      }
    }
  }
}


__device__ void BroadcastBottomEdgesInPlace(
    KernelArray arr,
    ArraySize2D tile_size
) {
  // Each thread is responsible for broadcasting from one column
  for (int local_col = 0; local_col < tile_size.num_cols; ++local_col) {
    int full_col = GetSharedMemCol(local_col, tile_size.num_cols);

    // === Step 1: Preload bottom row value for this column
    int source_row =
        GetSharedMemRow(tile_size.num_rows - 1, tile_size.num_rows);
    int idx_src = ArrayIndex1D(source_row, full_col, arr.size.num_cols);
    int edge_val = arr.d_address[idx_src];

    __syncthreads();

    // === Step 2: Broadcast down
    for (int block_row = threadIdx.y + 1; block_row < blockDim.y;
         ++block_row) {
      for (int downstream_row = 0; downstream_row < tile_size.num_rows;
           ++downstream_row) {
        int target_row = block_row * tile_size.num_rows + downstream_row;
        int idx_dst = ArrayIndex1D(target_row, full_col, arr.size.num_cols);
        arr.d_address[idx_dst] += edge_val;
      }
    }
  }
}
