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

__device__ int ArrayIndex1D(int row, int col, int num_cols) {
  return row * num_cols + col;
}

__device__ int METLocalColToFullArrayCol(int local_col, int tile_size_x) {
  return threadIdx.x * tile_size_x + local_col;
}

__device__ int METLocalRowToFullArrayRow(int local_row, int tile_size_y) {
  return threadIdx.y * tile_size_y + local_row;
}

__device__ void PrintMETTileContents(
    KernelArray array,
    ArraySize2D tile_size,
    int tile_row = 0,
    int tile_col = 0
) {
  if (threadIdx.x == tile_col && threadIdx.y == tile_row) {

    PrintThreadAndBlockIndices();
    for (int local_row = 0; local_row < tile_size.num_rows; ++local_row) {
      for (int local_col = 0; local_col < tile_size.num_cols; ++local_col) {
        int global_row =
            METLocalRowToFullArrayRow(local_row, tile_size.num_rows);
        int global_col =
            METLocalColToFullArrayCol(local_col, tile_size.num_cols);
        int index_1d =
            ArrayIndex1D(global_row, global_col, array.size.num_rows);
      }
      printf("\n");
    }
  }
}

__device__ void CopyMETTiledArray(
    KernelArray source,
    KernelArray dest,
    ArraySize2D tile_size
) {

  for (int local_row = 0; local_row < tile_size.num_rows; ++local_row) {
    for (int local_col = 0; local_col < tile_size.num_cols; ++local_col) {
      int full_array_col =
          METLocalColToFullArrayCol(local_col, tile_size.num_cols);
      int full_array_row =
          METLocalRowToFullArrayRow(local_row, tile_size.num_rows);

      int index_1d =
          ArrayIndex1D(full_array_row, full_array_col, source.size.num_cols);
      dest.d_address[index_1d] = source.d_address[index_1d];
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
    KernelArray arr,
    int other_row,
    int other_col,
    int cur_row,
    int cur_col
) {

  int cur_idx_1d = ArrayIndex1D(cur_row, cur_col, arr.size.num_cols);
  int other_idx_1d = ArrayIndex1D(other_row, other_col, arr.size.num_cols);
  arr.d_address[cur_idx_1d] += arr.d_address[other_idx_1d];
}

__device__ void ComputeLocalRowWisePrefixSums(
    KernelArray arr,
    ArraySize2D tile_size
) {
  __syncthreads();
  for (int local_col = 1; local_col < tile_size.num_cols; ++local_col) {
    for (int local_row = 0; local_row < tile_size.num_rows; ++local_row) {
      int full_array_col =
          METLocalColToFullArrayCol(local_col, tile_size.num_cols);
      int full_array_row =
          METLocalRowToFullArrayRow(local_row, tile_size.num_rows);
      int full_array_col_prev =
          METLocalColToFullArrayCol(local_col - 1, tile_size.num_cols);
      CombineElementInto(
          arr,
          full_array_row,
          full_array_col_prev,
          full_array_row,
          full_array_col
      );
    }
  }
}

__device__ void ComputeLocalColWisePrefixSums(
    KernelArray arr,
    ArraySize2D tile_size
) {

  for (int local_row = 1; local_row < tile_size.num_rows; ++local_row) {
    for (int local_col = 0; local_col < tile_size.num_cols; ++local_col) {
      int full_array_col =
          METLocalColToFullArrayCol(local_col, tile_size.num_cols);
      int full_array_row =
          METLocalRowToFullArrayRow(local_row, tile_size.num_rows);
      int full_array_row_prev =
          METLocalRowToFullArrayRow(local_row - 1, tile_size.num_rows);

      CombineElementInto(
          arr,
          full_array_row_prev,
          full_array_col,
          full_array_row,
          full_array_col
      );
    }
  }
}

__device__ void BroadcastRightEdges(
    KernelArray source_array,
    ArraySize2D tile_size,
    KernelArray dest_array
) {
  // iterate over each tile to right of cur_tile
  for (int block_col = threadIdx.x + 1; block_col < blockDim.x; ++block_col) {
    // iterate over each row of cur_tile
    for (int local_row = 0; local_row < tile_size.num_rows; ++local_row) {
      int full_array_row =
          METLocalRowToFullArrayRow(local_row, tile_size.num_rows);

      // iterate over each column of downstream tile
      for (int downstream_tile_col = 0;
           downstream_tile_col < tile_size.num_cols;
           ++downstream_tile_col) {

        int source_full_array_col = METLocalColToFullArrayCol(
            tile_size.num_cols - 1,
            tile_size.num_cols
        );
        int dest_full_array_col =
            block_col * tile_size.num_cols + downstream_tile_col;
        int dest_index_1d = ArrayIndex1D(
            full_array_row,
            dest_full_array_col,
            dest_array.size.num_cols
        );
        int source_index_1d = ArrayIndex1D(
            full_array_row,
            source_full_array_col,
            source_array.size.num_cols
        );
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
  for (int block_row = threadIdx.y + 1; block_row < blockDim.y; ++block_row) {
    // iterate over each col of cur_tile
    for (int local_col = 0; local_col < tile_size.num_cols; ++local_col) {
      // iterate over each row of downstream tile
      for (int downstream_tile_row = 0;
           downstream_tile_row < tile_size.num_rows;
           ++downstream_tile_row) {

        int full_array_col =
            METLocalColToFullArrayCol(local_col, tile_size.num_cols);
        int source_full_array_row = METLocalRowToFullArrayRow(
            tile_size.num_rows - 1,
            tile_size.num_rows
        );
        int dest_full_array_row =
            block_row * tile_size.num_rows + downstream_tile_row;

        int dest_index_1d = ArrayIndex1D(
            dest_full_array_row,
            full_array_col,
            dest_array.size.num_cols
        );
        int source_index_1d = ArrayIndex1D(
            source_full_array_row,
            full_array_col,
            source_array.size.num_cols
        );
        dest_array.d_address[dest_index_1d] +=
            source_array.d_address[source_index_1d];
      }
    }
  }
}

// hierarchichal helpers
