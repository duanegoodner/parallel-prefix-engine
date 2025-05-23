#pragma once

#include <cuda_runtime.h>

#include "cuda_prefix_sum/internal/kernel_array_view.cuh"

namespace row_scan_multi_block {
  __global__ void RowScanMultiBlockPhase1(
      const int *__restrict__ in_ptr,
      int *__restrict__ out_ptr,
      int *__restrict__ block_sums_ptr,
      ArraySize2D size,
      int chunk_size
  );

  __global__ void RowScanMultiBlockPhase2(
      int *__restrict__ out_ptr,
      const int *__restrict__ scanned_block_sums_ptr,
      ArraySize2D size,
      int chunk_size
  );

  __forceinline__ __device__ int RowIndex() { return blockIdx.y; }
  __forceinline__ __device__ int ColIndex() { return blockIdx.x; }
  __forceinline__ __device__ int NumChunks() { return gridDim.x; }
  __forceinline__ __device__ int ChunkStart(int chunk_size) {
    return ColIndex() * chunk_size;
  }
  __forceinline__ __device__ int ColOffset() { return threadIdx.x; }
  __forceinline__ __device__ int GlobalCol(int chunk_size) {
    return ChunkStart(chunk_size) + ColOffset();
  }

  // Device function to convert an inclusive scan to exclusive for a single
  // row.
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

} // namespace row_scan_multi_block