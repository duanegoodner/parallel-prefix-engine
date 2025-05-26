
#include <cuda_runtime.h>

#include "common/array_size_2d.hpp"
#include "cuda_prefix_sum/internal/kernel_array_view.cuh"
#include "cuda_prefix_sum/internal/col_scan_multi_block_kernel.cuh"

namespace col_scan_multi_block {

__global__ void ColScanMultiBlockPhase1(
    const int* __restrict__ in_ptr,
    int* __restrict__ out_ptr,
    int* __restrict__ block_sums_ptr,
    ArraySize2D size,
    int chunk_size
) {
  RowMajorKernelArrayViewConst in{in_ptr, size};
  RowMajorKernelArrayView out{out_ptr, size};

  extern __shared__ int temp[];
  RowMajorKernelArrayView shared_temp{temp, {static_cast<size_t>(chunk_size), 1}};

  const int col = ColIndex();
  const int global_row = GlobalRow(chunk_size);
  const int local_row = RowOffsetInChunk();

  // Load to shared memory
  if (global_row < size.num_rows) {
    shared_temp.At(local_row, 0) = in.At(global_row, col);
  } else {
    shared_temp.At(local_row, 0) = 0;
  }
  __syncthreads();

  // Inclusive scan along the column
  for (int offset = 1; offset < chunk_size; offset *= 2) {
    int val = (local_row >= offset) ? shared_temp.At(local_row - offset, 0) : 0;
    __syncthreads();
    shared_temp.At(local_row, 0) += val;
    __syncthreads();
  }

  // Convert to exclusive scan and write result
  if (global_row < size.num_rows) {
    ConvertInclusiveToExclusiveColumn(out, shared_temp, global_row, local_row);
  }

  // Write last value to block sums buffer
  if (local_row == chunk_size - 1 || global_row == size.num_rows - 1) {
    block_sums_ptr[col * gridDim.y + ChunkIndex()] = shared_temp.At(local_row, 0);
  }
}

__global__ void ColScanMultiBlockPhase2(
    int* __restrict__ out_ptr,
    const int* __restrict__ scanned_block_sums_ptr,
    ArraySize2D size,
    int chunk_size
) {
  RowMajorKernelArrayView out{out_ptr, size};
  RowMajorKernelArrayViewConst scanned_block_sums{
      scanned_block_sums_ptr,
      {static_cast<size_t>(gridDim.y), size.num_cols}
  };

  const int col = ColIndex();
  const int global_row = GlobalRow(chunk_size);
  const int chunk_id = ChunkIndex();

  if (chunk_id == 0 || global_row >= size.num_rows)
    return;

  int offset = scanned_block_sums.At(chunk_id, col);
  out.At(global_row, col) += offset;
}

}  // namespace col_scan_multi_block
