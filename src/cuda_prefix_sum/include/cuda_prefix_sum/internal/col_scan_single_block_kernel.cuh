#pragma once

#include <cuda_runtime.h>

#include "common/array_size_2d.hpp"
#include "cuda_prefix_sum/internal/kernel_array_view.cuh"

namespace col_scan_single_block {

__global__ void ColScanSingleBlockKernel(
    const int* __restrict__ in_ptr,
    int* __restrict__ out_ptr,
    ArraySize2D size
);

// Returns current column index (blockIdx.x corresponds to one column)
__forceinline__ __device__ int ColIndex() { return blockIdx.x; }

// Returns current row index within column (threadIdx.x steps down the column)
__forceinline__ __device__ int RowIndexInCol() { return threadIdx.x; }

// Load input from global to shared memory column-wise
__forceinline__ __device__ void LoadGlobalToSharedColumn(
    KernelArrayViewConst global_array,
    KernelArrayView shared_array
) {
  shared_array.At(RowIndexInCol(), 0) =
      global_array.At(RowIndexInCol(), ColIndex());
}

// Apply results of row-scanned right-edge prefixes
// __forceinline__ __device__ void 

// Inclusive Hillis-Steele scan down the column in shared memory
__forceinline__ __device__ void InclusiveScanDownColumn(
    KernelArrayView shared_temp,
    int num_rows
) {
  for (int offset = 1; offset < num_rows; offset *= 2) {
    int val = (RowIndexInCol() >= offset)
                  ? shared_temp.At(RowIndexInCol() - offset, 0)
                  : 0;
    __syncthreads();
    shared_temp.At(RowIndexInCol(), 0) += val;
    __syncthreads();
  }
}

// Convert inclusive to exclusive and write to output
__forceinline__ __device__ void ConvertToExclusiveAndStore(
    KernelArrayView shared_temp,
    KernelArrayView output
) {
  output.At(RowIndexInCol(), ColIndex()) =
      (RowIndexInCol() == 0) ? 0 : shared_temp.At(RowIndexInCol() - 1, 0);
}

}  // namespace col_scan_single_block
