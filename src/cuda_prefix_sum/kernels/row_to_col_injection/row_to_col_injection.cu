#include <cuda_runtime.h>

#include "common/array_size_2d.hpp"

#include "cuda_prefix_sum/internal/kernel_array_view.cuh"
#include "cuda_prefix_sum/internal/row_to_col_injection_kernel.cuh"

namespace row_to_col_injection {
  __global__ void RowToColInjection(
      int *__restrict__ col_array_ptr,
      ArraySize2D col_array_size,
      const int *__restrict__ row_prefix_array_ptr,
      ArraySize2D row_prefix_array_size,
      ArraySize2D tile_size
  ) {

    if (ColArrayRow() >= col_array_size.num_rows ||
        ColArrayCol() >= col_array_size.num_cols)
      return;

    RowMajorKernelArrayView col_array{col_array_ptr, col_array_size};
    RowMajorKernelArrayViewConst row_prefix_array{
        row_prefix_array_ptr,
        row_prefix_array_size
    };

    AdjustColArrayElement(row_prefix_array, col_array, tile_size);
  }

} // namespace row_to_col_injection