#include <cuda_runtime.h>

#include <cstdio>
#include <iostream>

// #include "cuda_prefix_sum/cuda_device_helpers.cuh"
#include "cuda_prefix_sum/cuda_prefix_sum_solver.cuh"
#include "cuda_prefix_sum/kernel_launch_params.hpp"

__device__ inline int index_2d(int x, int y, int tile_dim) {
    return x * tile_dim + y;
}

__global__ void PrefixSumKernelAccum(KernelLaunchParams params) {
    extern __shared__ int tile[];

    int tile_dim = params.tile_size.x;  // assuming square tiles
    int array_width = params.array.size.y;
    int array_height = params.array.size.x;

    int tid = threadIdx.x;
    int local_x = tid / tile_dim;
    int local_y = tid % tile_dim;

    if (local_x >= tile_dim || local_y >= tile_dim) return;

    int global_x = blockIdx.x * tile_dim + local_x;
    int global_y = blockIdx.y * tile_dim + local_y;
    if (global_x >= array_height || global_y >= array_width) return;

    int global_idx = global_x * array_width + global_y;

    // === Phase 1: Load to shared memory ===
    tile[index_2d(local_x, local_y, tile_dim)] = params.array.d_address[global_idx];
    __syncthreads();

    // === Phase 2: Row-wise scan ===
    int val = tile[index_2d(local_x, local_y, tile_dim)];
    for (int i = 1; i <= local_y; ++i) {
        val += tile[index_2d(local_x, local_y - i, tile_dim)];
    }
    __syncthreads();
    tile[index_2d(local_x, local_y, tile_dim)] = val;
    __syncthreads();

    // === Phase 3: Col-wise scan ===
    val = tile[index_2d(local_x, local_y, tile_dim)];
    for (int i = 1; i <= local_x; ++i) {
        val += tile[index_2d(local_x - i, local_y, tile_dim)];
    }
    __syncthreads();
    tile[index_2d(local_x, local_y, tile_dim)] = val;
    __syncthreads();

    // === Phase 4: Write back in-place ===
    params.array.d_address[global_idx] = tile[index_2d(local_x, local_y, tile_dim)];
}

void LaunchPrefixSumKernelAccum(KernelLaunchParams kernel_params) {
    int tile_dim = kernel_params.tile_size.x;
    int array_height = kernel_params.array.size.x;
    int array_width = kernel_params.array.size.y;

    dim3 grid(array_width / tile_dim, array_height / tile_dim);
    dim3 block(tile_dim * tile_dim);  // One thread per element

    size_t shared_mem_bytes = tile_dim * tile_dim * sizeof(int);

    PrefixSumKernelAccum<<<grid, block, shared_mem_bytes>>>(kernel_params);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}


