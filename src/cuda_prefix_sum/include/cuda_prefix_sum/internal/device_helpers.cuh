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
  const int local_row;
  const int local_col;
  ArraySize2D subtile_size;

  __forceinline__ __device__ int global_row() const {
    return blockIdx.y * blockDim.y * subtile_size.num_rows +
           threadIdx.y * subtile_size.num_rows + local_row;
  }

  __forceinline__ __device__ int global_col() const {
    return blockIdx.x * blockDim.x * subtile_size.num_cols +
           threadIdx.x * subtile_size.num_cols + local_col;
  }

  __forceinline__ __device__ int shared_row() const {
    return threadIdx.y * subtile_size.num_rows + local_row;
  }

  __forceinline__ __device__ int shared_col() const {
    return threadIdx.x * subtile_size.num_cols + local_col;
  }

  __forceinline__ __device__ int GlobalArrayIndex1D(int global_width) const {
    return ArrayIndex1D(global_row(), global_col(), global_width);
  }

  __forceinline__ __device__ int SharedArrayIndex1D() const {
    int tile_width = blockDim.x * subtile_size.num_cols;
    return ArrayIndex1D(shared_row(), shared_col(), tile_width);
  }
};

//
// Helper to construct an ElementCoords instance
//
__forceinline__ __device__ ElementCoords GetElementCoords(
    const int local_row,
    const int local_col,
    const ArraySize2D &subtile_size
) {
  return ElementCoords{local_row, local_col, subtile_size};
}

// Debug statement: Print thread and block indices
static __device__ void PrintThreadAndBlockIndices() {
  printf(
      "Block (%d, %d), Thread (%d, %d), Global Index: %d\n",
      blockIdx.x,
      blockIdx.y,
      threadIdx.x,
      threadIdx.y,
      threadIdx.x * blockDim.y + threadIdx.y
  );
}

static __device__ void PrintSubTileContents(
    KernelArray shared_array,
    const ArraySize2D sub_tile_size,
    const int tile_row_index = 0,
    const int tile_col_index = 0,
    const int sub_tile_row_index = 0,
    const int sub_tile_col_index = 0
) {
  if (blockIdx.x == tile_col_index && blockIdx.y == tile_row_index &&
      threadIdx.x == sub_tile_col_index && threadIdx.y == sub_tile_row_index) {

    PrintThreadAndBlockIndices();
    for (int local_row = 0; local_row < sub_tile_size.num_rows; ++local_row) {
      for (int local_col = 0; local_col < sub_tile_size.num_cols;
           ++local_col) {
        auto coords = GetElementCoords(local_row, local_col, sub_tile_size);
        auto shared_mem_index_1d = coords.SharedArrayIndex1D();
        printf(" %d", shared_array.d_address[shared_mem_index_1d]);
      }
      printf("\n");
    }
  }
}

static __device__ void CopyFromGlobalToShared(
    const KernelArray global_array,
    KernelArray shared_array,
    const ArraySize2D sub_tile_size
) {

  for (int local_row = 0; local_row < sub_tile_size.num_rows; ++local_row) {
    for (int local_col = 0; local_col < sub_tile_size.num_cols; ++local_col) {
      auto coords = GetElementCoords(local_row, local_col, sub_tile_size);
      auto global_index_1d =
          coords.GlobalArrayIndex1D(global_array.size.num_cols);
      auto shared_mem_index_1d = coords.SharedArrayIndex1D();
      shared_array.d_address[shared_mem_index_1d] =
          global_array.d_address[global_index_1d];
    }
  }
}

static __device__ void CopyFromSharedToGlobal(
    const KernelArray shared_array,
    KernelArray global_array,
    const ArraySize2D sub_tile_size
) {

  for (int local_row = 0; local_row < sub_tile_size.num_rows; ++local_row) {
    for (int local_col = 0; local_col < sub_tile_size.num_cols; ++local_col) {
      auto coords = GetElementCoords(local_row, local_col, sub_tile_size);
      auto global_index_1d =
          coords.GlobalArrayIndex1D(global_array.size.num_cols);
      auto shared_mem_index_1d = coords.SharedArrayIndex1D();
      global_array.d_address[global_index_1d] =
          shared_array.d_address[shared_mem_index_1d];
    }
  }
}

// Prints the contents of a thread's assigned subtile from shared memory.
// Only active for a specific blockIdx & threadIdx.
static __device__ void PrintKernelArray(KernelArray array, const char *label) {
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

static __device__ void CombineElementInto(
    KernelArray shared_array,
    const ElementCoords other_element,
    const ElementCoords cur_element
) {
  shared_array.d_address[cur_element.SharedArrayIndex1D()] +=
      shared_array.d_address[other_element.SharedArrayIndex1D()];
}

static __device__ void ComputeLocalRowWisePrefixSums(
    KernelArray shared_array,
    const ArraySize2D sub_tile_size
) {
  __syncthreads();
  for (int local_col = 1; local_col < sub_tile_size.num_cols; ++local_col) {
    for (int local_row = 0; local_row < sub_tile_size.num_rows; ++local_row) {
      auto cur_coords = GetElementCoords(local_row, local_col, sub_tile_size);
      auto other_coords =
          GetElementCoords(local_row, local_col - 1, sub_tile_size);
      CombineElementInto(shared_array, other_coords, cur_coords);
    }
  }
}

static __device__ void ComputeLocalColWisePrefixSums(
    KernelArray shared_array,
    const ArraySize2D sub_tile_size
) {

  for (int local_row = 1; local_row < sub_tile_size.num_rows; ++local_row) {
    for (int local_col = 0; local_col < sub_tile_size.num_cols; ++local_col) {
      auto cur_coords = GetElementCoords(local_row, local_col, sub_tile_size);
      auto other_coords =
          GetElementCoords(local_row - 1, local_col, sub_tile_size);
      CombineElementInto(shared_array, other_coords, cur_coords);
    }
  }
}

static __device__ void CollectRightEdges(
    KernelArray shared_array,
    const ArraySize2D sub_tile_size
) {

  // Each thread iterates over each row in its sub_tile
  for (int local_row = 0; local_row < sub_tile_size.num_rows; ++local_row) {
    // Index of full shared mem array row we are working in
    int shared_array_row = threadIdx.y * sub_tile_size.num_rows + local_row;

    // initialize accumulator register var to zero
    __syncthreads();
    int accumulator = 0;
    __syncthreads();

    // Each thread iterates over the sub-tiles (threads) upstream of it
    for (int src_thread_idx_x = 0; src_thread_idx_x < threadIdx.x;
         ++src_thread_idx_x) {
      // read current row's right edge value from upstream subtile and add to
      // accumulator
      int src_shared_array_col = src_thread_idx_x * sub_tile_size.num_cols +
                                 sub_tile_size.num_cols - 1;
      int src_shared_array_idx_1d = ArrayIndex1D(
          shared_array_row,
          src_shared_array_col,
          shared_array.size.num_cols
      );

      accumulator += shared_array.d_address[src_shared_array_idx_1d];
    }

    // wait until all threads have read all upstream edge vals in local_row
    __syncthreads();

    // iterate over each element in local row, and add accumulator val to it
    for (int local_col = 0; local_col < sub_tile_size.num_cols; ++local_col) {
      auto coords = GetElementCoords(local_row, local_col, sub_tile_size);
      shared_array.d_address[coords.SharedArrayIndex1D()] += accumulator;
    }
  }
}

static __device__ void CollectBottomEdges(
    KernelArray shared_array,
    const ArraySize2D sub_tile_size
) {
  // Each thread iterates over each col in its sub_tile
  for (int local_col = 0; local_col < sub_tile_size.num_cols; ++local_col) {
    // Index of full shared mem array col we are working in
    int shared_array_col = threadIdx.x * sub_tile_size.num_cols + local_col;

    // initialize accumulator register var to zero
    __syncthreads();
    int accumulator = 0;
    __syncthreads();

    // Each thread iterates over the sub-tiles (threads) upstream of it
    for (int src_thread_idx_y = 0; src_thread_idx_y < threadIdx.y;
         ++src_thread_idx_y) {

      // read current col's bottom edge value from upstream subtile and add to
      // accumulator
      int src_shared_array_row = src_thread_idx_y * sub_tile_size.num_rows +
                                 sub_tile_size.num_rows - 1;
      int src_shared_array_idx_1d = ArrayIndex1D(
          src_shared_array_row,
          shared_array_col,
          shared_array.size.num_cols
      );

      accumulator += shared_array.d_address[src_shared_array_idx_1d];
    }

    // wait until all threads have read all upstream edge vals in current col
    __syncthreads();

    // iterate over each element in local col, and add accumulator val to it
    for (int local_row = 0; local_row < sub_tile_size.num_rows; ++local_row) {
      auto coords = GetElementCoords(local_row, local_col, sub_tile_size);
      shared_array.d_address[coords.SharedArrayIndex1D()] += accumulator;
    }
  }
}

static __device__ void CopyTileRightEdgesToGlobalBuffer(
    KernelArray shared_array,
    int *right_edges_buffer,
    const ArraySize2D sub_tile_size
) {
  if (threadIdx.x == blockDim.x - 1) {
    int local_col = sub_tile_size.num_cols - 1;
    int buffer_col = blockIdx.x;
    for (int local_row = 0; local_row < sub_tile_size.num_rows; ++local_row) {
      auto coords = GetElementCoords(local_row, local_col, sub_tile_size);
      auto buffer_row = coords.global_row();
      auto buffer_index_1d = ArrayIndex1D(
          buffer_row,
          buffer_col,
          sub_tile_size.num_cols * blockDim.x * gridDim.x
      );
      right_edges_buffer[buffer_index_1d] =
          shared_array.d_address[coords.SharedArrayIndex1D()];
    }
  }
}

static __device__ void CopyTileBottomEdgesToGlobalBuffer(
    KernelArray shared_array,
    int *bottom_edges_buffer,
    const ArraySize2D sub_tile_size
) {
  if (threadIdx.y == blockDim.y - 1) {
    int local_row = sub_tile_size.num_rows - 1;
    int buffer_row = blockIdx.y;
    for (int local_col = 0; local_col < sub_tile_size.num_cols; ++local_col) {
      auto coords = GetElementCoords(local_row, local_col, sub_tile_size);
      auto buffer_col = coords.global_col();
      auto buffer_index_1d = ArrayIndex1D(
          buffer_row,
          buffer_col,
          sub_tile_size.num_cols * blockDim.x * gridDim.x
      );
      bottom_edges_buffer[buffer_index_1d] =
          shared_array.d_address[coords.SharedArrayIndex1D()];
    }
  }
}

static __device__ void CopyTileBottomEdgesToGlobalMemory(
    int *bottom_edges_buffer
) {}

static __device__ void ComputeSharedMemArrayPrefixSum(
    KernelArray shared_array,
    ArraySize2D sub_tile_size
) {
  ComputeLocalRowWisePrefixSums(shared_array, sub_tile_size);
  __syncthreads();

  ComputeLocalColWisePrefixSums(shared_array, sub_tile_size);
  __syncthreads();

  CollectRightEdges(shared_array, sub_tile_size);
  __syncthreads();

  CollectBottomEdges(shared_array, sub_tile_size);
}
