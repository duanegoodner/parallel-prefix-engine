#include "common/array_size_2d.hpp"

#include "cuda_prefix_sum/internal/kernel_array_view.cuh"
#include "cuda_prefix_sum/internal/row_scan_multi_block_kernel.cuh"

namespace row_scan_multi_block {
  // === Multi-block (chunked) scan for long rows ===
  // Phase 1: Local block scan (inclusive, then convert to exclusive)
 __global__ void RowScanMultiBlockPhase1(
    const int* __restrict__ in_ptr,
    int* __restrict__ out_ptr,
    int* __restrict__ block_sums_ptr,
    ArraySize2D size,
    int chunk_size
) {
  KernelArrayViewConst in{in_ptr, size};
  KernelArrayView out{out_ptr, size};

  extern __shared__ int temp[];
  KernelArrayView shared_temp{temp, {1, static_cast<size_t>(chunk_size)}};

  // int row = blockIdx.y;
  // int chunk_start = blockIdx.x * chunk_size;
  // int ColOffset() = threadIdx.x;
  // int global_col = ChunkStart(chunk_size) + ColOffset();

  // Load to shared
  if (GlobalCol(chunk_size) < size.num_cols) {
    shared_temp.At(0, ColOffset()) = in.At(RowIndex(), GlobalCol(chunk_size));
  } else {
    shared_temp.At(0, ColOffset()) = 0;
  }
  __syncthreads();

  // Inclusive scan in shared memory
  for (int offset = 1; offset < chunk_size; offset *= 2) {
    int val = (ColOffset() >= offset) ? shared_temp.At(0, ColOffset() - offset) : 0;
    __syncthreads();
    shared_temp.At(0, ColOffset()) += val;
    __syncthreads();
  }

  // Convert to exclusive scan and write result
  if (GlobalCol(chunk_size) < size.num_cols) {
    out.At(RowIndex(), GlobalCol(chunk_size)) = (ColOffset() == 0) ? 0 : shared_temp.At(0, ColOffset() - 1);
  }

  // Store full sum for this block
  if (ColOffset() == chunk_size - 1 || GlobalCol(chunk_size) == size.num_cols - 1) {
    block_sums_ptr[RowIndex() * gridDim.x + blockIdx.x] = shared_temp.At(0, ColOffset());
  }
}


  // Phase 2: Add scanned block sums to partial results
 __global__ void RowScanMultiBlockPhase2(
    int* __restrict__ out_ptr,
    const int* __restrict__ scanned_block_sums_ptr,
    ArraySize2D size,
    int chunk_size
) {
  KernelArrayView out{out_ptr, size};

  // int row = blockIdx.y;
  int chunk_id = blockIdx.x;
  // int ColOffset() = threadIdx.x;
  int chunk_start = chunk_id * chunk_size;
  // int global_col = chunk_start + ColOffset();

  if (chunk_id == 0 || GlobalCol(chunk_size) >= size.num_cols) return;

  int offset = scanned_block_sums_ptr[RowIndex() * gridDim.x + chunk_id];
  out.At(RowIndex(), GlobalCol(chunk_size)) += offset;
}


} // namespace row_scan_multi_block