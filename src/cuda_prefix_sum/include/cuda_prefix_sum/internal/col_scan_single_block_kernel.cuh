#pragma once

#include <cuda_runtime.h>

#include "common/array_size_2d.hpp"

#include "cuda_prefix_sum/internal/kernel_array_view.cuh"

namespace col_scan_single_block {

  __global__ void ColScanSingleBlockKernel(
      const int *__restrict__ input_ptr,
      int *__restrict__ output_ptr,
      ArraySize2D scan_array_size,
      const int *__restrict__ row_prefix_ptr,
      ArraySize2D row_prefix_array_size,
      ArraySize2D tile_size
  );

  // Returns current column index (blockIdx.x corresponds to one column)
  __forceinline__ __device__ int GlobalColIndex() { return blockIdx.x; }

  // Returns current row index within column (threadIdx.x steps down the
  // column)
  __forceinline__ __device__ int LocalRowIndex() { return threadIdx.x; }

  __forceinline__ __device__ int GlobalTileCol(ArraySize2D tile_size) {
    return blockIdx.x / tile_size.num_cols;
  }

  __forceinline__ __device__ int GlobalTileRow(ArraySize2D tile_size) {
    return threadIdx.x * tile_size.num_rows + tile_size.num_rows - 1;
  }

  // Load input from global to shared memory column-wise
  __forceinline__ __device__ void LoadColumnToShared(
      RowMajorKernelArrayViewConst global_array,
      RowMajorKernelArrayView shared_array
  ) {
    shared_array.At(LocalRowIndex(), 0) =
        global_array.At(LocalRowIndex(), GlobalColIndex());
  }

  // Apply results of row-scanned right-edge prefixes
  __forceinline__ __device__ void InjectRowPrefixAdjustment(
      RowMajorKernelArrayViewConst right_edge_ps_array,
      RowMajorKernelArrayView shared_col_view,
      ArraySize2D tile_size
  ) {

    shared_col_view.At(LocalRowIndex(), 0) +=
        right_edge_ps_array.At(
            GlobalTileRow(tile_size),
            GlobalTileCol(tile_size)
        );
  }

  // Inclusive Hillis-Steele scan down the column in shared memory
  __forceinline__ __device__ void InclusiveScanDownColumn(
      RowMajorKernelArrayView shared_col_view,
      int num_rows
  ) {
    for (int offset = 1; offset < num_rows; offset *= 2) {
      int val =
          (LocalRowIndex() >= offset)
              ? shared_col_view.At(LocalRowIndex() - offset, 0)
              : 0;
      __syncthreads();
      shared_col_view.At(LocalRowIndex(), 0) += val;
      __syncthreads();
    }
  }

  // Convert inclusive to exclusive and write to output
  __forceinline__ __device__ void StoreExclusiveResultToGlobal(
      RowMajorKernelArrayView shared_col_view,
      RowMajorKernelArrayView output
  ) {
    output.At(LocalRowIndex(), GlobalColIndex()) =
        (LocalRowIndex() == 0)
            ? 0
            : shared_col_view.At(LocalRowIndex() - 1, 0);
  }

} // namespace col_scan_single_block
