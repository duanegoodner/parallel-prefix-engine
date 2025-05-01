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

__global__ void PrefixSumKernelSingleElement(
    int *d_data
    // int rows_per_tile,
    // int cols_per_tile
) {
  // Print data on device global memory
  // PrintGlobalMemArray(d_data);

  // Declare dynamic shared memory
  extern __shared__ int shared_mem[];

  // Divide shared memory into two arrays
  int *arrayA = shared_mem;
  int *arrayB = shared_mem + blockDim.x * blockDim.y;

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
  arrayA[tx * blockDim.y + ty] = d_data[index];
  __syncthreads();

  // Debug statement: Print contents of arrayA after loading from global memory
  // PrintSharedMemoryArray(
  //     arrayA,
  //     "Contents of arrayA after loading from global memory"
  // );

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

void LaunchPrefixSumKernelSingleElement(
    int *d_data,
    int full_matrix_dim_x,
    int full_matrix_dim_y,
    cudaStream_t stream
) {

  dim3 blockDim(full_matrix_dim_x, full_matrix_dim_y);
  dim3 gridDim(1, 1); // Single block for now

  PrefixSumKernelSingleElement<<<gridDim, blockDim, 0, stream>>>(d_data);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }

  cudaDeviceSynchronize();
}