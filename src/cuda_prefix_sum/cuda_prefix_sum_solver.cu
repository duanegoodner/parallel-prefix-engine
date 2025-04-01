// cuda_prefix_sum_solver.cu
//
// Defines the CUDA kernel and launch function for performing 2D prefix sum.
// This file contains only GPU-side logic and is compiled by NVCC.

#include <cuda_runtime.h>

#include <cstdio>
#include <iostream>

#include "cuda_prefix_sum/cuda_prefix_sum_solver.cuh" // Host-side declaration


__device__ void PrintSharedMemoryArray(const int *array, int tile_dim, const char *label) {
  // Only thread (0, 0) in the block prints the array to avoid excessive output
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    printf("%s:\n", label);
    for (int i = 0; i < tile_dim; ++i) {
      for (int j = 0; j < tile_dim; ++j) {
        printf("%d ", array[i * tile_dim + j]);
      }
      printf("\n");
    }
  }
  __syncthreads(); // Ensure all threads reach this point
}


__global__ void PrefixSumKernel(int *d_data, int tile_dim) {
  // Declare dynamic shared memory
  extern __shared__ int shared_mem[];

  // Divide shared memory into two arrays
  int *arrayA = shared_mem;
  int *arrayB = shared_mem + tile_dim * tile_dim;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int index = ty * tile_dim + tx;

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
  arrayA[ty * tile_dim + tx] = d_data[index];
  __syncthreads();

  // Debug statement: Print contents of arrayA after loading from global memory
  PrintSharedMemoryArray(arrayA, tile_dim, "Contents of arrayA after loading from global memory");

  // === Phase 2: Row-wise prefix sum into arrayB ===
  int sum = 0;
  for (int i = 0; i <= tx; ++i) {
    sum += arrayA[ty * tile_dim + i];
  }
  arrayB[ty * tile_dim + tx] = sum;

  __syncthreads();

  // Debug: Print contents of arrayB after row-wise prefix sum
  PrintSharedMemoryArray(arrayB, tile_dim, "Contents of arrayB after row-wise prefix sum");

  // === Phase 3: Column-wise prefix sum (over arrayB) into arrayA ===
  sum = 0;
  for (int i = 0; i <= ty; ++i) {
    sum += arrayB[i * tile_dim + tx];
  }
  arrayA[ty * tile_dim + tx] = sum;

  __syncthreads();

  // Debug: Print contents of arrayA after column-wise prefix sum
  PrintSharedMemoryArray(arrayA, tile_dim, "Contents of arrayA after column-wise prefix sum");

  // === Phase 4: Write final result back to global memory ===
  d_data[index] = arrayA[ty * tile_dim + tx];
}

void LaunchPrefixSumKernel(int *d_data, int tile_dim, cudaStream_t stream) {
  dim3 blockDim(tile_dim, tile_dim);
  dim3 gridDim(1, 1); // Single block for now

  PrefixSumKernel<<<gridDim, blockDim, 0, stream>>>(d_data, tile_dim);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }

  cudaDeviceSynchronize();
}
