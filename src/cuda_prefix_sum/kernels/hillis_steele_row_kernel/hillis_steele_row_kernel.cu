#include "cuda_prefix_sum/internal/device_helpers.cuh"
#include "cuda_prefix_sum/internal/hillis_steele_row_kernel.cuh"

namespace dh = device_helpers;

// === Single-block exclusive scan (Hillis-Steele) for a single row ===
__global__ void RowWiseScanSingleBlock(
    const int *__restrict__ in_ptr,
    int *__restrict__ out_ptr,
    ArraySize2D size
) {
  KernelArrayViewConst in{in_ptr, size};
  KernelArrayView out{out_ptr, size};

  extern __shared__ int temp[];
  KernelArrayView shared_temp{temp, {1, size.num_cols}};

  if (dh::blockrow::ColIndex() >= size.num_cols)
    return;

  // Load input to shared memory
  dh::blockrow::LoadGlobalArrayToSharedArray(in, shared_temp, 0);
  __syncthreads();

  // Inclusive Hillis-Steele scan
  dh::blockrow::InclusiveHillsSteeleScan(shared_temp, size.num_cols);

  // Convert to exclusive scan
  dh::blockrow::ConvertInclusiveToExclusive(shared_temp, 0, out);
}

// === Multi-block (chunked) scan for long rows ===
// Phase 1: Local block scan (inclusive, then convert to exclusive)
__global__ void RowWiseScanMultiBlockPhase1(
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
    shared_temp.At(0, col_offset) =
        in.At(dh::blockrow::chunks::RowIndex(), global_col);
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
    dh::ConvertInclusiveToExclusiveRow(
        out,
        shared_temp,
        dh::blockrow::chunks::RowIndex(),
        col_offset
    );
  }

  // Store full sum for this block
  if (col_offset == chunk_size - 1 || global_col == size.num_cols - 1) {
    block_sums_view.At(
        dh::blockrow::chunks::RowIndex(),
        dh::blockrow::chunks::RowIndex()
    ) = shared_temp.At(0, col_offset);
  }
}

// Phase 2: Add scanned block sums to partial results
__global__ void RowWiseScanMultiBlockPhase2(
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
