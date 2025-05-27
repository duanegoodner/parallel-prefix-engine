#include <cuda_runtime.h>

#include "common/array_size_2d.hpp"

#include "cuda_prefix_sum/internal/kernel_array_view.cuh"

namespace row_to_col_injection {
  __global__ void RowToColInjection(
      int *__restrict__ col_array_ptr,
      ArraySize2D col_array_size,
      const int *__restrict__ row_prefix_array_ptr,
      ArraySize2D row_prefix_array_size,
      ArraySize2D tile_size
  );

  __forceinline__ __device__ int ColArrayRow() {
    return blockIdx.y *blockDim.y + threadIdx.y;
  }

  __forceinline__ __device__ int ColArrayCol() { return blockIdx.x; }

  __forceinline__ __device__ int GlobalTileRow(ArraySize2D tile_size) {
    return ColArrayRow() * tile_size.num_rows + tile_size.num_rows - 1;
  }

  __forceinline__ __device__ int GlobalTileCol(ArraySize2D tile_size) {
    return blockIdx.x / tile_size.num_cols;
  }

  __forceinline__ __device__ void AdjustColArrayElement(
      RowMajorKernelArrayViewConst row_prefix_array,
      RowMajorKernelArrayView col_array,
      ArraySize2D tile_size
  ) {
    col_array.At(ColArrayRow(), ColArrayCol()) += row_prefix_array.At(
        GlobalTileRow(tile_size),
        GlobalTileCol(tile_size)
    );
  }

} // namespace row_to_col_injection
