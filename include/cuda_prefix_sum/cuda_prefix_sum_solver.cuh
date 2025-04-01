// cuda_prefix_sum_solver.cuh
#pragma once
#include <cuda_runtime.h>

void LaunchPrefixSumKernel(int *d_data, int tile_dim, cudaStream_t stream = 0);
