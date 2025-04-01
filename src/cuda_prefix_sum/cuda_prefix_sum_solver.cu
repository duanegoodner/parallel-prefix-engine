// cuda_prefix_sum_solver.cu
//
// Defines the CUDA kernel and launch function for performing 2D prefix sum.
// This file contains only GPU-side logic and is compiled by NVCC.

#include <cuda_runtime.h>

#include <cstdio>

#include "cuda_prefix_sum/cuda_prefix_sum_solver.cuh" // Host-side declaration

constexpr int TILE_DIM = 32;

__global__ void PrefixSumKernel(int *d_data) {
  __shared__ int arrayA[TILE_DIM][TILE_DIM];
  __shared__ int arrayB[TILE_DIM][TILE_DIM];

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int index = ty * TILE_DIM + tx;

  // === Phase 1: Load input from global memory to shared memory ===
  arrayA[ty][tx] = d_data[index];

  __syncthreads();

  // === Phase 2: Row-wise prefix sum into arrayB ===
  int sum = 0;
  for (int i = 0; i <= tx; ++i) {
    sum += arrayA[ty][i];
  }
  arrayB[ty][tx] = sum;

  __syncthreads();

  // === Phase 3: Column-wise prefix sum (over arrayB) into arrayA ===
  sum = 0;
  for (int i = 0; i <= ty; ++i) {
    sum += arrayB[i][tx];
  }
  arrayA[ty][tx] = sum;

  __syncthreads();

  // === Phase 4: Write final result back to global memory ===
  d_data[index] = arrayA[ty][tx];
}

void LaunchPrefixSumKernel(int *d_data, int tile_dim, cudaStream_t stream) {
  dim3 blockDim(TILE_DIM, TILE_DIM);
  dim3 gridDim(1, 1); // Single block for now

  PrefixSumKernel<<<gridDim, blockDim, 0, stream>>>(d_data);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }

  cudaDeviceSynchronize();
}
