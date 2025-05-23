#pragma once

#include <cuda_runtime.h>

#include "common/array_size_2d.hpp"

#include "cuda_prefix_sum/internal/kernel_array_view.cuh"

namespace row_scan_single_block {
  __global__ void RowScanSingleBlockKernel(
      const int *__restrict__ in_ptr,
      int *__restrict__ out_ptr,
      ArraySize2D size
  );

  static __forceinline__ __device__ int RowIndex() { return blockIdx.x; }
  static __forceinline__ __device__ int ColIndex() { return threadIdx.x; }

  static __forceinline__ __device__ void LoadGlobalArrayToSharedArray(
      KernelArrayViewConst global_array,
      KernelArrayView shared_array,
      int shared_array_row
  ) {
    shared_array.At(shared_array_row, ColIndex()) =
        global_array.At(RowIndex(), ColIndex());
  }

  static __forceinline__ __device__ void InclusiveHillsSteeleScan(
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

  static __forceinline__ __device__ void ConvertInclusiveToExclusive(
      KernelArrayView shared_temp,
      int shared_temp_row,
      KernelArrayView output
  ) {
    output.At(RowIndex(), ColIndex()) =
        (ColIndex() == 0) ? 0
                          : shared_temp.At(shared_temp_row, ColIndex() - 1);
  }
};