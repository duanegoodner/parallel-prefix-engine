// cuda_prefix_sum_solver.cu
//
// Defines the CUDA kernel and launch function for performing 2D prefix sum.
// This file contains only GPU-side logic and is compiled by NVCC.

#include <cuda_runtime.h>

#include <cstdio>
#include <iostream>

#include "cuda_prefix_sum/cuda_device_helpers.cuh"
#include "cuda_prefix_sum/cuda_prefix_sum_solver.cuh"
#include "cuda_prefix_sum/kernel_launch_params.hpp"

__global__ void PrefixSumKernel(
    int *d_data,
    KernelLaunchParams params
) {
  // Declare dynamic shared memory
  extern __shared__ int shared_mem[];

  // Divide shared memory into two arrays
  int *arrayA = shared_mem;
  int *arrayB =
      shared_mem + params.full_matrix_dim_x * params.full_matrix_dim_y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int index = tx * blockDim.y + ty;


  // === Phase 1: Load input from global memory to shared memory ===

  LoadFromGlobalToSharedMemory(d_data, arrayA, params);

  __syncthreads();

  // Debug statement: Print contents of arrayA after loading from global memory
  PrintArray(
      arrayA,
      params.full_matrix_dim_x,
      params.full_matrix_dim_y,
      "Contents of arrayA after loading from global memory"
  );

  // === Phase 2: Row-wise prefix sum within each tile of arrayA ===
  for (int tile_col = 1; tile_col < params.tile_size_y; tile_col++) {
    for (int tile_row = 0; tile_row < params.tile_size_x; ++tile_row) {
      int full_matrix_x =
          ArrayIndexX(tile_row, tile_col, params.tile_size_x);
      int full_matrix_y =
          ArrayIndexY(tile_row, tile_col, params.tile_size_y);
      int full_matrix_y_prev =
          ArrayIndexY(tile_row, tile_col - 1, params.tile_size_y);
      CombineElementInto(
          arrayA,
          params.full_matrix_dim_x,
          params.full_matrix_dim_y,
          full_matrix_x,
          full_matrix_y_prev,
          full_matrix_x,
          full_matrix_y
      );
    }
  }

  __syncthreads();
  // Debug statement: Print contents of arrayA after row-wise prefix sum
  PrintArray(
      arrayA,
      params.full_matrix_dim_x,
      params.full_matrix_dim_y,
      "Contents of arrayA after row-wise prefix sum"
  );

  // === Phase 3: Column-wise prefix sum within each tile of arrayA ===
  for (int tile_row = 1; tile_row < params.tile_size_x; tile_row++) {
    for (int tile_col = 0; tile_col < params.tile_size_y; ++tile_col) {
      int full_matrix_x = tx * params.tile_size_x + tile_row;
      int full_matrix_y = ty * params.tile_size_y + tile_col;
      int full_matrix_x_prev = tx * params.tile_size_x + tile_row - 1;
      arrayA[full_matrix_x * params.full_matrix_dim_y + full_matrix_y] +=
          arrayA
              [full_matrix_x_prev * params.full_matrix_dim_y + full_matrix_y];
    }
  }

  __syncthreads();
  // Debug statement: Print contents of arrayA after column-wise prefix sum
  PrintArray(
      arrayA,
      params.full_matrix_dim_x,
      params.full_matrix_dim_y,
      "Contents of arrayA after column-wise prefix sum"
  );

  // === Phase 4: Compute/write final result into arrayB ===

  // Extract right edges of upstream tiles
  for (int upstream_tile_col = 0; upstream_tile_col < ty;
       ++upstream_tile_col) {
    int upstream_tile_full_matrix_col_idx =
        upstream_tile_col * params.tile_size_y + params.tile_size_y - 1;
    for (int tile_row = 0; tile_row < params.tile_size_x; ++tile_row) {
      int full_matrix_row_idx = tx * params.tile_size_x + tile_row;
      int edge_val = arrayA
          [full_matrix_row_idx * params.full_matrix_dim_y +
           upstream_tile_full_matrix_col_idx];
      for (int tile_col = 0; tile_col < params.tile_size_y; ++tile_col) {
        int full_matrix_x = tx * params.tile_size_x + tile_row;
        int full_matrix_y = ty * params.tile_size_y + tile_col;
        arrayB[full_matrix_x * params.full_matrix_dim_y + full_matrix_y] =
            arrayA[full_matrix_x * params.full_matrix_dim_y + full_matrix_y] +
            edge_val;
      }
    }
  }

  __syncthreads();
  // Debug statement: Print contents of arrayB after adding upstream right
  // edges
  PrintArray(
      arrayB,
      params.full_matrix_dim_x,
      params.full_matrix_dim_y,
      "Contents of arrayB extracting/adding right edges of upstream tiles"
  );

  // Extrat bottom edges of upstream tiles

  for (int tile_row = 0; tile_row < params.tile_size_x; ++tile_row) {
    for (int tile_col = 0; tile_col < params.tile_size_y; ++tile_col) {
      int full_matrix_x = tx * params.tile_size_x + tile_row;
      int full_matrix_y = ty * params.tile_size_y + tile_col;
    }
  }

  // === Phase 2: Row-wise prefix sum into arrayB ===
  int sum = 0;
  for (int col = 0; col <= ty; ++col) {
    // TODO: implement op
    sum += arrayA[tx * blockDim.y + col];
  }

  arrayB[tx * blockDim.y + ty] = sum;

  __syncthreads();

  // PrintSharedMemoryArray(
  //     arrayB,
  //     "Contents of arrayB after row-wise prefix sum"
  // );

  // === Phase 3: Column-wise prefix sum (over arrayB) into arrayA ===
  sum = 0;
  for (int row = 0; row <= tx; ++row) {
    // TODO: implement op
    sum += arrayB[row * blockDim.y + ty];
  }
  arrayA[tx * blockDim.y + ty] = sum;

  __syncthreads();

  // Debug: Print contents of arrayA after column-wise prefix sum
  // PrintSharedMemoryArray(
  //     arrayA,
  //     "Contents of arrayA after column-wise prefix sum"
  // );

  // === Phase 4: Write final result back to global memory ===
  d_data[index] = arrayA[tx * blockDim.y + ty];

  // PrintGlobalMemArray(d_data);
}

void LaunchPrefixSumKernel(
    int *d_data,
    KernelLaunchParams kernel_params,
    // int full_matrix_dim_x,
    // int full_matrix_dim_y,
    // int tile_size_x,
    // int tile_size_y,
    cudaStream_t stream
) {

  int num_tiles_x = kernel_params.full_matrix_dim_x / kernel_params.tile_size_x;
  int num_tiles_y = kernel_params.full_matrix_dim_y / kernel_params.tile_size_y;

  // dim3 blockDim(full_matrix_dim_x, full_matrix_dim_y);
  dim3 blockDim(num_tiles_x, num_tiles_y);
  dim3 gridDim(1, 1); // Single block for now

  PrefixSumKernel<<<gridDim, blockDim, 0, stream>>>(
      d_data,
      kernel_params
      // full_matrix_dim_x,
      // full_matrix_dim_y,
      // tile_size_x,
      // tile_size_y
  );

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }

  cudaDeviceSynchronize();
}
