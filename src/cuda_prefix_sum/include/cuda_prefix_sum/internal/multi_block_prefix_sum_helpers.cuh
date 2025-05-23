#pragma once

#include "cuda_prefix_sum/internal/kernel_array.hpp"
#include "common/array_size_2d.hpp"

namespace multi_block_prefix_sum {

inline size_t ComputeNumChunks(size_t dim_length, int chunk_size) {
  return (dim_length + chunk_size - 1) / chunk_size;
}

inline dim3 ComputeGridDim(bool row_major, size_t num_chunks, size_t secondary_dim) {
  return row_major ? dim3(num_chunks, secondary_dim)
                   : dim3(secondary_dim, num_chunks);
}

inline size_t SharedMemBytes(int chunk_size) {
  return chunk_size * sizeof(int);
}

template <typename Phase1Kernel>
inline void LaunchPhase1(
    Phase1Kernel phase1_kernel,
    const int* d_input,
    int* d_output,
    int* d_block_sums,
    ArraySize2D size,
    dim3 grid,
    dim3 block,
    size_t shared_mem_bytes,
    int chunk_size
) {
  phase1_kernel<<<grid, block, shared_mem_bytes>>>(
      d_input, d_output, d_block_sums, size, chunk_size);
}

template <typename Phase2Kernel>
inline void LaunchPhase2(
    Phase2Kernel phase2_kernel,
    int* d_output,
    const int* d_scanned_block_sums,
    ArraySize2D size,
    dim3 grid,
    dim3 block,
    int chunk_size
) {
  phase2_kernel<<<grid, block>>>(
      d_output, d_scanned_block_sums, size, chunk_size);
}

template <typename Phase1Kernel, typename Phase2Kernel, typename RecursiveLauncher>
inline void Launch(
    const int* d_input,
    int* d_output,
    ArraySize2D size,
    int chunk_size,
    bool row_major,
    Phase1Kernel phase1_kernel,
    Phase2Kernel phase2_kernel,
    RecursiveLauncher recursive_launcher
) {
  size_t primary_dim = row_major ? size.num_cols : size.num_rows;
  size_t secondary_dim = row_major ? size.num_rows : size.num_cols;
  size_t num_chunks = ComputeNumChunks(primary_dim, chunk_size);

  dim3 grid = ComputeGridDim(row_major, num_chunks, secondary_dim);
  dim3 block(chunk_size);
  size_t shared_mem_bytes = SharedMemBytes(chunk_size);

  int* d_block_sums;
  cudaMalloc(&d_block_sums, sizeof(int) * secondary_dim * num_chunks);

  LaunchPhase1(
      phase1_kernel,
      d_input,
      d_output,
      d_block_sums,
      size,
      grid,
      block,
      shared_mem_bytes,
      chunk_size);

  int* d_scanned_block_sums;
  cudaMalloc(&d_scanned_block_sums, sizeof(int) * secondary_dim * num_chunks);

  ArraySize2D block_sum_size{secondary_dim, num_chunks};
  recursive_launcher(d_block_sums, d_scanned_block_sums, block_sum_size);

  LaunchPhase2(
      phase2_kernel,
      d_output,
      d_scanned_block_sums,
      size,
      grid,
      block,
      chunk_size);

  cudaFree(d_block_sums);
  cudaFree(d_scanned_block_sums);
}

}  // namespace multi_block_prefix_sum
