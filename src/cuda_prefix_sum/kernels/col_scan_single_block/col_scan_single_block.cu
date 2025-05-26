#include <cuda_runtime.h>

#include "common/array_size_2d.hpp"

#include "cuda_prefix_sum/internal/col_scan_single_block_kernel.cuh"
#include "cuda_prefix_sum/internal/kernel_array_view.cuh"

namespace col_scan_single_block {

  __global__ void ColScanSingleBlockKernel(
      const int *__restrict__ input_ptr,
      int *__restrict__ output_ptr,
      ArraySize2D scan_array_size,
      const int *__restrict__ row_prefix_ptr,
      ArraySize2D row_prefix_array_size,
      ArraySize2D tile_size
  ) {
    // One thread per row, one block per column
    if (LocalRowIndex() >= scan_array_size.num_rows)
      return;

    KernelArrayViewConst input_view{input_ptr, scan_array_size};
    KernelArrayView output_view{output_ptr, scan_array_size};
    KernelArrayViewConst row_prefix_view{
        row_prefix_ptr,
        row_prefix_array_size
    };

    extern __shared__ int shared_col_buffer_ptr[];
    KernelArrayView shared_col_view{
        shared_col_buffer_ptr,
        {scan_array_size.num_rows, 1}
    };

    LoadColumnToShared(input_view, shared_col_view);
    __syncthreads();

    InjectRowPrefixAdjustment(
        row_prefix_view,
        shared_col_view,
        tile_size
    );
    __syncthreads();

    InclusiveScanDownColumn(
        shared_col_view,
        scan_array_size.num_rows
    );
    __syncthreads();

    StoreExclusiveResultToGlobal(shared_col_view, output_view);
  }

} // namespace col_scan_single_block
