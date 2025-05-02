#include <cuda_runtime.h>
#include "cuda_prefix_sum/cuda_prefix_sum_solver.cuh"

namespace config {
    constexpr int TileDim = 32;
    constexpr int TilePitch = TileDim + 1;
    // constexpr int NumElems = TileDim * TileDim;
}

// Kernel implementation
__global__ void PrefixSumTile32x32(const int* input, int* output) {
    __shared__ int tile[config::TileDim * config::TilePitch];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int global_idx = ty * config::TileDim + tx;
    int smem_idx = ty * config::TilePitch + tx;

    tile[smem_idx] = input[global_idx];
    __syncthreads();

    int val = tile[smem_idx];
    #pragma unroll
    for (int offset = 1; offset < config::TileDim; offset *= 2) {
        int n = __shfl_up_sync(0xffffffff, val, offset);
        if (tx >= offset) val += n;
    }
    tile[smem_idx] = val;
    __syncthreads();

    val = tile[smem_idx];
    #pragma unroll
    for (int offset = 1; offset < config::TileDim; offset *= 2) {
        int n = __shfl_up_sync(0xffffffff, val, offset);
        if (ty >= offset) val += n;
    }
    tile[smem_idx] = val;
    __syncthreads();

    output[global_idx] = tile[smem_idx];
}

// Kernel launcher
void LaunchPrefixSumWarpKernel(const int* d_input, int* d_output) {
    dim3 block(config::TileDim, config::TileDim);
    PrefixSumTile32x32<<<1, block>>>(d_input, d_output);
}
