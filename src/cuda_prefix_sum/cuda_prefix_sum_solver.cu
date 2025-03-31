// cuda_prefix_sum_solver.cu
//
// Defines the CUDA kernel and launch function for performing 2D prefix sum.
// This file contains only GPU-side logic and is compiled by NVCC.

#include <cuda_runtime.h>
#include <iostream>

__global__ void PrefixSumKernel(int* data, int tile_dim) {
  int row = threadIdx.y;
  int col = threadIdx.x;

  int idx = row * tile_dim + col;

  if (row < tile_dim && col < tile_dim) {
    // Toy operation for now: add +1 to each element
    data[idx] += 1;
  }
}

extern "C" void LaunchPrefixSumKernel(int* d_data, int tile_dim, cudaStream_t stream) {
  dim3 threads(tile_dim, tile_dim);  // one thread per matrix element
  dim3 blocks(1, 1);  // one block for now

  PrefixSumKernel<<<blocks, threads, 0, stream>>>(d_data, tile_dim);
  cudaDeviceSynchronize();
}
