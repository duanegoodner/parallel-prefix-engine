#include "common/array_size_2d.hpp"

#include "cuda_prefix_sum/internal/kernel_array_view.cuh"
#include "cuda_prefix_sum/internal/row_scan_multi_block_kernel.cuh"

namespace row_scan_multi_block {
  // === Multi-block (chunked) scan for long rows ===
  // Phase 1: Local block scan (inclusive, then convert to exclusive)
  __global__ void RowScanMultiBlockPhase1(
      const int *__restrict__ in_ptr,
      int *__restrict__ out_ptr,
      int *__restrict__ block_sums_ptr,
      ArraySize2D size,
      int chunk_size
  ) {
    //   int row = blockIdx.y;
    int num_chunks = gridDim.x;
    int chunk_start = blockIdx.x * chunk_size;
    int col_offset = threadIdx.x;
    int global_col = chunk_start + col_offset;

    KernelArrayViewConst in{in_ptr, size};
    KernelArrayView out{out_ptr, size};

    KernelArrayView block_sums_view{
        block_sums_ptr,
        {size.num_rows, static_cast<size_t>(num_chunks)}
    };

    extern __shared__ int temp[];
    KernelArrayView shared_temp{temp, {1, static_cast<size_t>(chunk_size)}};

    // Load to shared
    if (global_col < size.num_cols) {
      shared_temp.At(0, col_offset) = in.At(RowIndex(), global_col);
    } else {
      shared_temp.At(0, col_offset) = 0;
    }
    __syncthreads();

    // Inclusive scan in shared memory
    for (int offset = 1; offset < chunk_size; offset *= 2) {
      int val =
          (col_offset >= offset) ? shared_temp.At(0, col_offset - offset) : 0;
      __syncthreads();
      shared_temp.At(0, col_offset) += val;
      __syncthreads();
    }

    // Convert to exclusive scan and write result
    if (global_col < size.num_cols) {
      ConvertInclusiveToExclusiveRow(out, shared_temp, RowIndex(), col_offset);
    }

    // Store full sum for this block
    if (col_offset == chunk_size - 1 || global_col == size.num_cols - 1) {
      block_sums_view.At(RowIndex(), RowIndex()) =
          shared_temp.At(0, col_offset);
    }
  }

  // Phase 2: Add scanned block sums to partial results
  __global__ void RowScanMultiBlockPhase2(
      int *__restrict__ out_ptr,
      const int *__restrict__ scanned_block_sums_ptr,
      ArraySize2D size,
      int chunk_size
  ) {
    KernelArrayView out{out_ptr, size};

    int row = blockIdx.y;
    int chunk_id = blockIdx.x;
    int col_offset = threadIdx.x;
    int chunk_start = chunk_id * chunk_size;
    int global_col = chunk_start + col_offset;

    if (chunk_id == 0 || global_col >= size.num_cols)
      return;

    int offset = scanned_block_sums_ptr[row * gridDim.x + chunk_id];
    out.At(row, global_col) += offset;
  }

} // namespace row_scan_multi_block