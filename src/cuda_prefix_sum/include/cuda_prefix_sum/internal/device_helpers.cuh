#pragma once

#include "common/array_size_2d.hpp"

#include "cuda_prefix_sum/internal/kernel_array_view.cuh"
#include "cuda_prefix_sum/internal/kernel_launch_params.hpp"





namespace device_helpers {

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


namespace blockrow {
__forceinline__ __device__ int RowIndex() { return blockIdx.x; }
__forceinline__ __device__ int ColIndex() { return threadIdx.x; }

__forceinline__ __device__ void LoadGlobalArrayToSharedArray(
    KernelArrayViewConst global_array,
    KernelArrayView shared_array,
    int shared_array_row
) {
  shared_array.At(shared_array_row, ColIndex()) =
      global_array.At(RowIndex(), ColIndex());
}

__forceinline__ __device__ void InclusiveHillsSteeleScan(
    KernelArrayView shared_temp,
    int num_cols
) {
  for (int offset = 1; offset < num_cols; offset *= 2) {
    int val =
        (ColIndex() >= offset) ? shared_temp.At(0, ColIndex() - offset) : 0;
    __syncthreads();
    shared_temp.At(0, ColIndex()) += val;
    __syncthreads();
  }
}

__forceinline__ __device__ void ConvertInclusiveToExclusive(
    KernelArrayView shared_temp,
    int shared_temp_row,
    KernelArrayView output
) {
  output.At(RowIndex(), ColIndex()) =
      (ColIndex() == 0) ? 0 : shared_temp.At(shared_temp_row, ColIndex() - 1);
}

namespace chunks {
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

} // namespace chunks
} // namespace blockrow

namespace blockcol {
__forceinline__ __device__ int RowIndex() { return threadIdx.x; }
__forceinline__ __device__ int ColIndex() { return blockIdx.x; }
namespace chunks {}
} // namespace blockcol
} // namespace device_helpers
