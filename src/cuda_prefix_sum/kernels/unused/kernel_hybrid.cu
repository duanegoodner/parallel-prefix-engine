#include <cuda_runtime.h>

// #include "cuda_prefix_sum/cuda_prefix_sum_solver.cuh"
#include "cuda_prefix_sum/kernel_launch_params.hpp"

__device__ inline int index_2d(int row, int col, int width) {
  return row * width + col;
}

__global__ void PrefixSumHybridKernel(
    const int *input,
    int *output,
    ArraySize2D array_size,
    ArraySize2D tile_size
) {
  int subtile_rows = tile_size.num_rows;
  int subtile_cols = tile_size.num_cols;

  int block_dim_y = blockDim.y;
  int block_dim_x = blockDim.x;

  int TILE_DIM = subtile_rows * block_dim_y;

  // Use dynamic shared memory
  extern __shared__ int shared_mem[];
  int *tile_shared = shared_mem;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int tile_origin_x = blockIdx.y * TILE_DIM;
  int tile_origin_y = blockIdx.x * TILE_DIM;

  // === Phase 1: Load global to shared ===
  for (int i = 0; i < TILE_DIM; i += block_dim_y) {
    for (int j = 0; j < TILE_DIM; j += block_dim_x) {
      int row = tile_origin_x + ty + i;
      int col = tile_origin_y + tx + j;

      if (row < array_size.num_rows && col < array_size.num_cols) {
        int global_idx = row * array_size.num_cols + col;
        tile_shared[index_2d(ty + i, tx + j, TILE_DIM)] = input[global_idx];
      }
    }
  }
  __syncthreads();

  // === Phase 2: Row-wise scan
  for (int i = 0; i < TILE_DIM; i += block_dim_y) {
    int row = ty + i;
    if (row < TILE_DIM) {
      int val = tile_shared[index_2d(row, tx, TILE_DIM)];
      for (int k = 1; k <= tx; ++k) {
        val += tile_shared[index_2d(row, tx - k, TILE_DIM)];
      }
      tile_shared[index_2d(row, tx, TILE_DIM)] = val;
    }
  }
  __syncthreads();

  // === Phase 3: Col-wise scan
  for (int j = 0; j < TILE_DIM; j += block_dim_x) {
    int col = tx + j;
    if (col < TILE_DIM) {
      int val = tile_shared[index_2d(ty, col, TILE_DIM)];
      for (int k = 1; k <= ty; ++k) {
        val += tile_shared[index_2d(ty - k, col, TILE_DIM)];
      }
      tile_shared[index_2d(ty, col, TILE_DIM)] = val;
    }
  }
  __syncthreads();

  // === Phase 4: Store back to global memory
  for (int i = 0; i < TILE_DIM; i += block_dim_y) {
    for (int j = 0; j < TILE_DIM; j += block_dim_x) {
      int row = tile_origin_x + ty + i;
      int col = tile_origin_y + tx + j;

      if (row < array_size.num_rows && col < array_size.num_cols) {
        int global_idx = row * array_size.num_cols + col;
        output[global_idx] = tile_shared[index_2d(ty + i, tx + j, TILE_DIM)];
      }
    }
  }
}

void LaunchPrefixSumKernelHybrid(KernelLaunchParams params) {
  int subtile_rows = params.tile_size.num_rows;
  int subtile_cols = params.tile_size.num_cols;

  // Threads per block
  int block_dim_x = params.array.size.num_cols / subtile_cols;
  int block_dim_y = params.array.size.num_rows / subtile_rows;

  int TILE_DIM = subtile_rows * block_dim_y;

  dim3 block(block_dim_x, block_dim_y);
  dim3 grid(
      (params.array.size.num_cols + TILE_DIM - 1) / TILE_DIM,
      (params.array.size.num_rows + TILE_DIM - 1) / TILE_DIM
  );

  size_t shared_mem_bytes = TILE_DIM * TILE_DIM * sizeof(int);

  PrefixSumHybridKernel<<<grid, block, shared_mem_bytes>>>(
      params.array.d_address,
      params.array.d_address,
      params.array.size,
      params.tile_size
  );

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }

  cudaDeviceSynchronize();
  cudaError_t sync_err = cudaGetLastError();
  if (sync_err != cudaSuccess) {
    fprintf(stderr, "CUDA sync error: %s\n", cudaGetErrorString(sync_err));
  }
}
