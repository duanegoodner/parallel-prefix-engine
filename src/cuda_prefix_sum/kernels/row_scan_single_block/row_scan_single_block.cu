#include <cuda_runtime.h>

#include "common/array_size_2d.hpp"

#include "cuda_prefix_sum/internal/kernel_array_view.cuh"
#include "cuda_prefix_sum/internal/row_scan_single_block_kernel.cuh"

namespace row_scan_single_block {

__global__ void RowScanSingleBlockKernel(
    const int *__restrict__ in_ptr,
    int *__restrict__ out_ptr,
    ArraySize2D size
) {
  KernelArrayViewConst in{in_ptr, size};
  KernelArrayView out{out_ptr, size};

  extern __shared__ int temp[];
  KernelArrayView shared_temp{temp, {1, size.num_cols}};

  if (ColIndex() >= size.num_cols) return;

  LoadRowToShared(in, shared_temp, RowIndex());
  __syncthreads();

  InclusiveScanHillisSteele(shared_temp, size.num_cols);
  __syncthreads();

  ConvertToExclusiveScan(shared_temp, out, RowIndex());
}

} // namespace row_scan_single_block
