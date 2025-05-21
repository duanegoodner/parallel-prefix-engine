#include "cuda_prefix_sum/internal/device_helpers.cuh"

// === Single-block exclusive scan (Hillis-Steele) for a single row ===
__global__ void RowWiseScanSingleBlock(
    const int *__restrict__ in,
    int *__restrict__ out,
    // int num_cols
    ArraySize2D size
) {
  extern __shared__ int temp[];
  int row = blockIdx.x;
  int tid = threadIdx.x;

  if (tid >= size.num_cols)
    return;

  // Load input to shared memory
  temp[tid] = in[row * size.num_cols + tid];
  __syncthreads();

  // Inclusive Hillis-Steele scan
  for (int offset = 1; offset < size.num_cols; offset *= 2) {
    int val = 0;
    if (tid >= offset)
      val = temp[tid - offset];
    __syncthreads();
    temp[tid] += val;
    __syncthreads();
  }

  // Convert to exclusive scan
  if (tid == 0) {
    out[row * size.num_cols + tid] = 0;
  } else if (tid < size.num_cols) {
    out[row * size.num_cols + tid] = temp[tid - 1];
  }
}

// === Multi-block (chunked) scan for long rows ===
// Phase 1: Local block scan (inclusive, then convert to exclusive)
__global__ void RowWiseScanMultiBlockPhase1(
    const int *__restrict__ in,
    int *__restrict__ out,
    int *__restrict__ block_sums,
    ArraySize2D size,
    int chunk_size
) {
  extern __shared__ int temp[];
  int row = blockIdx.y;
  int chunk_start = blockIdx.x * chunk_size;
  int tid = threadIdx.x;
  int global_idx = row * size.num_cols + chunk_start + tid;

  // Load input
  if (chunk_start + tid < size.num_cols)
    temp[tid] = in[global_idx];
  else
    temp[tid] = 0;

  __syncthreads();

  // Inclusive scan within block
  for (int offset = 1; offset < chunk_size; offset *= 2) {
    int val = 0;
    if (tid >= offset)
      val = temp[tid - offset];
    __syncthreads();
    temp[tid] += val;
    __syncthreads();
  }

  // Convert to exclusive and store result
  if (chunk_start + tid < size.num_cols) {
    if (tid == 0) {
      out[global_idx] = 0;
    } else {
      out[global_idx] = temp[tid - 1];
    }
  }

  // Store inclusive block sum (last value of inclusive scan)
  if (tid == chunk_size - 1 || chunk_start + tid == size.num_cols - 1) {
    block_sums[row * gridDim.x + blockIdx.x] = temp[tid];
  }
}


// Phase 2: Add scanned block sums to partial results
__global__ void RowWiseScanMultiBlockPhase2(
    int *__restrict__ out,
    const int *__restrict__ scanned_block_sums,
    // int num_cols,
    ArraySize2D size,
    int chunk_size
) {
  int row = blockIdx.y;
  int chunk_id = blockIdx.x;
  int tid = threadIdx.x;

  int chunk_start = chunk_id * chunk_size;
  int global_idx = row * size.num_cols + chunk_start + tid;

  int offset = scanned_block_sums[row * gridDim.x + chunk_id];
  if (chunk_id > 0 && chunk_start + tid < size.num_cols) {
    out[global_idx] += offset;
  }
}