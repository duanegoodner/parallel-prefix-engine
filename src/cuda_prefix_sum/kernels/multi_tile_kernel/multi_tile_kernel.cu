

#include <cuda_runtime.h>

#include "cuda_prefix_sum/internal/device_helpers.cuh"
#include "cuda_prefix_sum/internal/kernel_array.hpp"
#include "cuda_prefix_sum/internal/kernel_launch_params.hpp"
#include "cuda_prefix_sum/internal/multi_tile_kernel.cuh"

__global__ void FirstPass(
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