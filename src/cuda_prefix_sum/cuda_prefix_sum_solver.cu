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
    int rows_per_tile,
    int cols_per_tile,
    const char *label
) {
  // Only thread (0, 0) in the block prints the array to avoid excessive output
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    printf("%s:\n", label);
    for (int i = 0; i < rows_per_tile; ++i) {
      for (int j = 0; j < cols_per_tile; ++j) {
        printf("%d ", array[i * rows_per_tile + j]);
      }
      printf("\n");
    }
  }
  __syncthreads(); // Ensure all threads reach this point
}

__global__ void PrefixSumKernel(
    int *d_data,
    int rows_per_tile,
    int cols_per_tile
) {
  // Declare dynamic shared memory
  extern __shared__ int shared_mem[];

  // Divide shared memory into two arrays
  int *arrayA = shared_mem;
  int *arrayB = shared_mem + rows_per_tile * cols_per_tile;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int index = ty * rows_per_tile + tx;

  // Debug statement: Print thread and block indices
  printf(
      "Block (%d, %d), Thread (%d, %d), Global Index: %d\n",
      blockIdx.x,
      blockIdx.y,
      tx,
      ty,
      index
  );

  // === Phase 1: Load input from global memory to shared memory ===
  arrayA[ty * rows_per_tile + tx] = d_data[index];
  __syncthreads();

  // Debug statement: Print contents of arrayA after loading from global memory
  PrintSharedMemoryArray(
      arrayA,
      rows_per_tile,
      cols_per_tile,
      "Contents of arrayA after loading from global memory"
  );

  // === Phase 2: Row-wise prefix sum into arrayB ===
  int sum = 0;
  for (int i = 0; i <= tx; ++i) {
    sum += arrayA[ty * rows_per_tile + i];
  }
  arrayB[ty * rows_per_tile + tx] = sum;

  __syncthreads();

  // Debug: Print contents of arrayB after row-wise prefix sum
  PrintSharedMemoryArray(
      arrayB,
      rows_per_tile,
      cols_per_tile,
      "Contents of arrayB after row-wise prefix sum"
  );

  // === Phase 3: Column-wise prefix sum (over arrayB) into arrayA ===
  sum = 0;
  for (int i = 0; i <= ty; ++i) {
    sum += arrayB[i * cols_per_tile + tx];
  }
  arrayA[ty * rows_per_tile + tx] = sum;

  __syncthreads();

  // Debug: Print contents of arrayA after column-wise prefix sum
  PrintSharedMemoryArray(
      arrayA,
      rows_per_tile,
      cols_per_tile,
      "Contents of arrayA after column-wise prefix sum"
  );

  // === Phase 4: Write final result back to global memory ===
  d_data[index] = arrayA[ty * rows_per_tile + tx];
}

void LaunchPrefixSumKernel(
    int *d_data,
    int rows_per_block,
    int cols_per_block,
    int blocks_per_row,
    int blocks_per_col,
    cudaStream_t stream
) {
  std::cout << "Launching PrefixSumKernel with dimensions: "
            << blocks_per_row << "x" << blocks_per_col
            << " and block size: " << rows_per_block << "x" << cols_per_block
            << std::endl;
  dim3 blockDim(rows_per_block, cols_per_block);
  dim3 gridDim(1, 1); // Single block for now

  PrefixSumKernel<<<gridDim, blockDim, 0, stream>>>(
      d_data,
      rows_per_block,
      cols_per_block
  );

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }

  cudaDeviceSynchronize();
}
