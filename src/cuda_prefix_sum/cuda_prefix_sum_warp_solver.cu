#include <cuda_runtime.h>

#include "cuda_prefix_sum/cuda_prefix_sum_solver.cuh"

namespace config {
constexpr int TileDim = 32;
constexpr int TilePitch = TileDim + 1;
// constexpr int NumElems = TileDim * TileDim;
} // namespace config

__global__ void PrefixSumKernelWarp(int *data) {
  __shared__ int tile[config::TileDim * config::TilePitch];

  // Original thread indices
  int tx = threadIdx.x; // horizontal (column index in row-pass)
  int ty = threadIdx.y; // vertical (row index in row-pass)

  int global_idx = ty * config::TileDim + tx;
  int smem_idx = ty * config::TilePitch + tx;

  // Load global memory into shared memory
  tile[smem_idx] = data[global_idx];
  __syncthreads();

  // --- Row-wise inclusive scan (1 warp per row) ---
  int val = tile[smem_idx];
#pragma unroll
  for (int offset = 1; offset < config::TileDim; offset *= 2) {
    int n = __shfl_up_sync(0xffffffff, val, offset);
    if (tx >= offset)
      val += n;
  }
  tile[smem_idx] = val;
  __syncthreads();

  // --- Column-wise inclusive scan (1 warp per column) ---
  int row = threadIdx.x; // Remap: threads step down a column
  int col = threadIdx.y; // Each warp owns one column

  int smem_col_idx = row * config::TilePitch + col;
  val = tile[smem_col_idx];
#pragma unroll
  for (int offset = 1; offset < config::TileDim; offset *= 2) {
    int n = __shfl_up_sync(0xffffffff, val, offset);
    if (row >= offset)
      val += n;
  }
  tile[smem_col_idx] = val;
  __syncthreads();

  // Restore thread indices to write final result
  int final_idx = ty * config::TilePitch + tx;
  data[global_idx] = tile[final_idx];
}

__global__ void PrefixSumKernelWarpNaive(int *data) {
  __shared__ int tile[config::TileDim * config::TilePitch];

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int global_idx = ty * config::TileDim + tx;
  int smem_idx = ty * config::TilePitch + tx;

  tile[smem_idx] = data[global_idx];
  __syncthreads();

  // --- Naive warp-based row prefix sum ---
  int val = tile[smem_idx];
  for (int i = 0; i < tx; ++i) {
    int neighbor = __shfl_sync(0xffffffff, val, i);
    val += neighbor;
  }
  tile[smem_idx] = val;
  __syncthreads();

  // --- Naive warp-based column prefix sum ---
  // Reinterpret threads: each warp owns one column
  int row = tx; // this thread will walk down the column
  int col = ty;

  int smem_col_idx = row * config::TilePitch + col;
  val = tile[smem_col_idx];

  for (int i = 0; i < row; ++i) {
    int neighbor = __shfl_sync(0xffffffff, val, i);
    val += neighbor;
  }
  tile[smem_col_idx] = val;
  __syncthreads();

  data[global_idx] = tile[smem_idx];
}

// Kernel launcher
void LaunchPrefixSumKernelWarp(KernelLaunchParams params) {
  dim3 block(config::TileDim, config::TileDim);
  PrefixSumKernelWarp<<<1, block>>>(params.array.d_address);
}

void LaunchPrefixSumKernelWarpNaive(KernelLaunchParams params) {
  dim3 block(config::TileDim, config::TileDim);
  PrefixSumKernelWarp<<<1, block>>>(params.array.d_address);
}
