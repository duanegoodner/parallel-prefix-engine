#pragma once

#include "common/array_size_2d.hpp"

#include "cuda_prefix_sum/internal/kernel_array_view.cuh"
#include "cuda_prefix_sum/internal/kernel_launch_params.hpp"

//
// Row-major 1D index calculation
//
// __forceinline__ __device__ int ArrayIndex1D(int row, int col, int num_cols) {
//   return row * num_cols + col;
// }

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
    KernelArrayView shared_array,
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
        printf(" %d", shared_array.At(local_row, local_col));
      }
      printf("\n");
    }
  }
}

static __device__ void CopyFromGlobalToShared(
    const KernelArrayView global_array,
    KernelArrayView shared_array,
    const ArraySize2D sub_tile_size
) {

  for (int local_row = 0; local_row < sub_tile_size.num_rows; ++local_row) {
    for (int local_col = 0; local_col < sub_tile_size.num_cols; ++local_col) {
      auto coords = GetElementCoords(local_row, local_col, sub_tile_size);
      shared_array.At(coords.shared_row(), coords.shared_col()) =
          global_array.At(coords.global_row(), coords.global_col());
    }
  }
}

static __device__ void CopyFromSharedToGlobal(
    const KernelArrayView shared_array,
    KernelArrayView global_array,
    const ArraySize2D sub_tile_size
) {

  for (int local_row = 0; local_row < sub_tile_size.num_rows; ++local_row) {
    for (int local_col = 0; local_col < sub_tile_size.num_cols; ++local_col) {
      auto coords = GetElementCoords(local_row, local_col, sub_tile_size);
      global_array.At(coords.global_row(), coords.global_col()) =
          shared_array.At(coords.shared_row(), coords.shared_col());
    }
  }
}

// Prints the contents of a thread's assigned subtile from shared memory.
// Only active for a specific blockIdx & threadIdx.
static __device__ void PrintKernelArrayView(
    KernelArrayView array,
    const char *label
) {
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    printf("%s:\n", label);
    for (int row = 0; row < array.size.num_rows; ++row) {
      for (int col = 0; col < array.size.num_cols; ++col) {
        printf(
            "%d\t",
            array.At(row, col)
            // array.d_address[ArrayIndex1D(row, col, array.size.num_cols)]
        );
      }
      printf("\n");
    }
    printf("\n");
  }
}

static __device__ void CombineElementInto(
    KernelArrayView shared_array,
    const ElementCoords other_element,
    const ElementCoords cur_element
) {
  shared_array.At(cur_element.shared_row(), cur_element.shared_col()) +=
      shared_array.At(other_element.shared_row(), other_element.shared_col());
}

static __device__ void ComputeLocalRowWisePrefixSums(
    KernelArrayView shared_array,
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
    KernelArrayView shared_array,
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
    KernelArrayView shared_array,
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
      accumulator += shared_array.At(shared_array_row, src_shared_array_col);
    }

    // wait until all threads have read all upstream edge vals in local_row
    __syncthreads();

    // iterate over each element in local row, and add accumulator val to it
    for (int local_col = 0; local_col < sub_tile_size.num_cols; ++local_col) {
      auto coords = GetElementCoords(local_row, local_col, sub_tile_size);
      shared_array.At(coords.shared_row(), coords.shared_col()) += accumulator;
    }
  }
}

static __device__ void CollectBottomEdges(
    KernelArrayView shared_array,
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
      accumulator += shared_array.At(src_shared_array_row, shared_array_col);
    }

    // wait until all threads have read all upstream edge vals in current col
    __syncthreads();

    // iterate over each element in local col, and add accumulator val to it
    for (int local_row = 0; local_row < sub_tile_size.num_rows; ++local_row) {
      auto coords = GetElementCoords(local_row, local_col, sub_tile_size);
      shared_array.At(coords.shared_row(), coords.shared_col()) += accumulator;
      // shared_array.d_address[coords.SharedArrayIndex1D()] += accumulator;
    }
  }
}

static __device__ void CopyTileRightEdgesToGlobalBuffer(
    KernelArrayView shared_array,
    KernelArrayView right_edges_buffer,
    const ArraySize2D sub_tile_size
) {
  if (threadIdx.x == blockDim.x - 1) {
    int local_col = sub_tile_size.num_cols - 1;
    int buffer_col = blockIdx.x;
    for (int local_row = 0; local_row < sub_tile_size.num_rows; ++local_row) {
      auto coords = GetElementCoords(local_row, local_col, sub_tile_size);
      auto buffer_row = coords.global_row();
      right_edges_buffer.At(buffer_row, buffer_col) =
          shared_array.At(coords.shared_row(), coords.shared_col());
    }
  }
}

static __device__ void CopyTileBottomEdgesToGlobalBuffer(
    KernelArrayView shared_array,
    KernelArrayView bottom_edges_buffer,
    const ArraySize2D sub_tile_size
) {
  if (threadIdx.y == blockDim.y - 1) {
    int local_row = sub_tile_size.num_rows - 1;
    int buffer_row = blockIdx.y;
    for (int local_col = 0; local_col < sub_tile_size.num_cols; ++local_col) {
      auto coords = GetElementCoords(local_row, local_col, sub_tile_size);
      auto buffer_col = coords.global_col();
      bottom_edges_buffer.At(buffer_row, buffer_col) =
          shared_array.At(coords.shared_row(), coords.shared_col());
    }
  }
}



static __device__ void ComputeSharedMemArrayPrefixSum(
    KernelArrayView shared_array,
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

// Device function to convert an inclusive scan to exclusive for a single row.
// The caller is responsible for ensuring shared_temp contains valid data.
__forceinline__ __device__ void ConvertInclusiveToExclusiveRow(
    KernelArrayView out,
    KernelArrayView shared_temp,
    int row_index,
    int col_index
) {
  if (col_index == 0) {
    out.At(row_index, col_index) = 0;
  } else {
    out.At(row_index, col_index) = shared_temp.At(row_index, col_index - 1);
  }
}

