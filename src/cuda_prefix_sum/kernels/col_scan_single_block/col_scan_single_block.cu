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
    if (ColScanArrayRow() >= scan_array_size.num_rows)
      return;

    KernelArrayViewConst input_array_view{input_ptr, scan_array_size};
    KernelArrayView output_array_view{output_ptr, scan_array_size};
    KernelArrayViewConst rowscan_result{
        row_prefix_ptr,
        row_prefix_array_size
    };

    extern __shared__ int shared_mem_single_col_ptr[];
    KernelArrayView shared_mem_single_col_view{
        shared_mem_single_col_ptr,
        {scan_array_size.num_rows, 1}
    };

    LoadColFromGlobalToSharedMem(input_array_view, shared_mem_single_col_view);
    __syncthreads();

    ApplyRowScanResult(
        rowscan_result,
        shared_mem_single_col_view,
        tile_size
    );
    __syncthreads();

    InclusiveScanDownColumn(
        shared_mem_single_col_view,
        scan_array_size.num_rows
    );
    __syncthreads();

    ConvertToExclusiveAndStore(shared_mem_single_col_view, output_array_view);
  }

} // namespace col_scan_single_block
