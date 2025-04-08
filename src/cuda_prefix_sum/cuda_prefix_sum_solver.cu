// cuda_prefix_sum_solver.cu
//
// Defines the CUDA kernel and launch function for performing 2D prefix sum.
// This file contains only GPU-side logic and is compiled by NVCC.

#include <cuda_runtime.h>

#include <cstdio>
#include <iostream>

#include "cuda_prefix_sum/cuda_prefix_sum_solver.cuh" // Host-side declaration

__device__ void PrintSharedMemoryArray(
    const int *array,
    // int rows_per_tile,
    // int cols_per_tile,
    const char *label
) {
  // Only thread (0, 0) in the block prints the array to avoid excessive output
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    printf("%s:\n", label);
    for (int row = 0; row < blockDim.x; ++row) {
      for (int col = 0; col < blockDim.y; ++col) {
        printf("%d\t", array[row * blockDim.y + col]);
      }
      printf("\n");
    }
  }
  __syncthreads(); // Ensure all threads reach this point
}

__device__ void PrintSharedMemoryArrayNew(
    const int *array,
    int full_matrix_dim_x,
    int full_matrix_dim_y,
    const char *label
) {
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    printf("%s:\n", label);
    for (int row = 0; row < full_matrix_dim_x; ++row) {
      for (int col = 0; col < full_matrix_dim_y; ++col) {
        printf("%d\t", array[row * full_matrix_dim_y + col]);
      }
      printf("\n");
    }
  }
}

__device__ void PrintGlobalMemArray(int *d_data) {
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    printf("Data on Device Global Memory:\n");
    for (int row = 0; row < blockDim.x; ++row) {
      for (int col = 0; col < blockDim.y; ++col) {
        printf("%d\t", d_data[row * blockDim.y + col]);
      }
      printf("\n");
    }
  }
}

__global__ void PrefixSumKernel(
    int *d_data,
    int full_matrix_dim_x,
    int full_matrix_dim_y,
    int tile_dim_x,
    int tile_dim_y
) {
  // Print data on device global memory
  // PrintGlobalMemArray(d_data);

  // Declare dynamic shared memory
  extern __shared__ int shared_mem[];

  // Divide shared memory into two arrays
  int *arrayA = shared_mem;
  int *arrayB = shared_mem + full_matrix_dim_x * full_matrix_dim_y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int index = tx * blockDim.y + ty;

  // Debug statement: Print thread and block indices
  // printf(
  //     "Block (%d, %d), Thread (%d, %d), Global Index: %d\n",
  //     blockIdx.x,
  //     blockIdx.y,
  //     tx,
  //     ty,
  //     index
  // );

  // === Phase 1: Load input from global memory to shared memory ===
  for (int tile_row = 0; tile_row < tile_dim_x; ++tile_row) {
    for (int tile_col = 0; tile_col < tile_dim_y; ++tile_col) {
      int full_matrix_x = tx * tile_dim_x + tile_row;
      int full_matrix_y = ty * tile_dim_y + tile_col;
      arrayA[full_matrix_x * full_matrix_dim_y + full_matrix_y] =
          d_data[full_matrix_x * full_matrix_dim_y + full_matrix_y];
    }
  }

  __syncthreads();

  // Debug statement: Print contents of arrayA after loading from global memory
  PrintSharedMemoryArrayNew(
      arrayA,
      full_matrix_dim_x,
      full_matrix_dim_y,
      "Contents of arrayA after loading from global memory"
  );

  // === Phase 2: Row-wise prefix sum within each tile of arrayA ===
  for (int tile_col = 1; tile_col < tile_dim_y; tile_col++) {
    for (int tile_row = 0; tile_row < tile_dim_x; ++tile_row) {
      int full_matrix_x = tx * tile_dim_x + tile_row;
      int full_matrix_y = ty * tile_dim_y + tile_col;
      int full_matrix_y_prev = ty * tile_dim_y + tile_col - 1;
      arrayA[full_matrix_x * full_matrix_dim_y + full_matrix_y] +=
          arrayA[full_matrix_x * full_matrix_dim_y + full_matrix_y_prev];
    }
  }

  __syncthreads();
  // Debug statement: Print contents of arrayA after row-wise prefix sum
  PrintSharedMemoryArrayNew(
      arrayA,
      full_matrix_dim_x,
      full_matrix_dim_y,
      "Contents of arrayA after row-wise prefix sum"
  );

  // === Phase 3: Column-wise prefix sum within each tile of arrayA ===
  for (int tile_row = 1; tile_row < tile_dim_x; tile_row++) {
    for (int tile_col = 0; tile_col < tile_dim_y; ++tile_col) {
      int full_matrix_x = tx * tile_dim_x + tile_row;
      int full_matrix_y = ty * tile_dim_y + tile_col;
      int full_matrix_x_prev =  tx * tile_dim_x + tile_row - 1;
      arrayA[full_matrix_x * full_matrix_dim_y + full_matrix_y] +=
          arrayA[full_matrix_x_prev * full_matrix_dim_y + full_matrix_y];
    }
  }
 

  __syncthreads();
  // Debug statement: Print contents of arrayA after column-wise prefix sum
  PrintSharedMemoryArrayNew(
      arrayA,
      full_matrix_dim_x,
      full_matrix_dim_y,
      "Contents of arrayA after column-wise prefix sum"
  );

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
    int full_matrix_dim_x,
    int full_matrix_dim_y,
    int tile_dim_x,
    int tile_dim_y,
    cudaStream_t stream
) {

  int num_tiles_x = full_matrix_dim_x / tile_dim_x;
  int num_tiles_y = full_matrix_dim_y / tile_dim_y;

  // dim3 blockDim(full_matrix_dim_x, full_matrix_dim_y);
  dim3 blockDim(num_tiles_x, num_tiles_y);
  dim3 gridDim(1, 1); // Single block for now

  PrefixSumKernel<<<gridDim, blockDim, 0, stream>>>(
      d_data,
      full_matrix_dim_x,
      full_matrix_dim_y,
      tile_dim_x,
      tile_dim_y
  );

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }

  cudaDeviceSynchronize();
}
