#include <cuda_runtime.h>

#include "common/array_size_2d.hpp"

#include "cuda_prefix_sum/internal/col_scan_single_block_kernel.cuh"
#include "cuda_prefix_sum/internal/kernel_array_view.cuh"

namespace col_scan_single_block {

  __global__ void ColScanSingleBlockKernel(
      const int *__restrict__ in_ptr,
      int *__restrict__ out_ptr,
      ArraySize2D bottom_edge_buffer_size,
      const int *__restrict__ rowscan_result_ptr,
      ArraySize2D right_edge_buffer_size,
      ArraySize2D tile_size
  ) {
    // One thread per row, one block per column
    if (RowIndexInCol() >= bottom_edge_buffer_size.num_rows)
      return;

    KernelArrayViewConst in{in_ptr, bottom_edge_buffer_size};
    KernelArrayView out{out_ptr, bottom_edge_buffer_size};
    KernelArrayViewConst rowscan_result{
        rowscan_result_ptr,
        right_edge_buffer_size
    };

    extern __shared__ int temp[];
    KernelArrayView shared_temp{temp, {bottom_edge_buffer_size.num_rows, 1}};

    LoadGlobalToSharedColumn(in, shared_temp);
    __syncthreads();

    ApplyRightEdgeResult(rowscan_result, shared_temp, tile_size);
    __syncthreads();

    InclusiveScanDownColumn(shared_temp, bottom_edge_buffer_size.num_rows);
    __syncthreads();

    ConvertToExclusiveAndStore(shared_temp, out);
  }

} // namespace col_scan_single_block
