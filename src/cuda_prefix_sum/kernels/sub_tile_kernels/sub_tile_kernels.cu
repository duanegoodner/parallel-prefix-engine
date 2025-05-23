#include <cuda_runtime.h>

#include <cstdio>
#include <iostream>

#include "cuda_prefix_sum/internal/device_helpers.cuh"
#include "cuda_prefix_sum/internal/kernel_launch_params.hpp"
#include "cuda_prefix_sum/internal/sub_tile_kernels.cuh"

namespace subtile_kernels {
__global__ void SingleTileKernel(
    // int *d_data,
    KernelLaunchParams params
) {
  // Declare dynamic shared memory
  extern __shared__ int shared_mem[];

  // __syncthreads();

  // Declare shared memory
  KernelArrayView shared_array{
      .d_address = shared_mem,
      .size = params.tile_size
  };
  // __syncthreads();

  // === Phase 1: Load input from global memory to shared memory ===
  CopyFromGlobalToShared(params.array, shared_array, params.sub_tile_size);
  __syncthreads();

  // === Phase 2: Compute 2D prefix sum on shared mem array ===
  ComputeSharedMemArrayPrefixSum(shared_array, params.sub_tile_size);
  __syncthreads();

  // === Phase 3: Write final result back to global memory ===
  CopyFromSharedToGlobal(shared_array, params.array, params.sub_tile_size);
}

__global__ void MultiTileKernel(
    KernelLaunchParams params,
    KernelArrayView right_edges_buffer,
    KernelArrayView bottom_edges_buffer
) {
  extern __shared__ int shared_mem[];

  KernelArrayView shared_array{
      .d_address = shared_mem,
      .size = params.tile_size
  };

  CopyFromGlobalToShared(params.array, shared_array, params.sub_tile_size);
  __syncthreads();

  ComputeSharedMemArrayPrefixSum(shared_array, params.sub_tile_size);
  __syncthreads();

  CopyFromSharedToGlobal(shared_array, params.array, params.sub_tile_size);
  __syncthreads();

  CopyTileRightEdgesToGlobalBuffer(
      shared_array,
      right_edges_buffer,
      params.sub_tile_size
  );
  CopyTileBottomEdgesToGlobalBuffer(
      shared_array,
      bottom_edges_buffer,
      params.sub_tile_size
  );
  __syncthreads();
}

} // namespace subtile_kernels
