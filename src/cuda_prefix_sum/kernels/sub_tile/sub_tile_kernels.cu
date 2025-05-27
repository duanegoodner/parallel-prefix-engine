#include <cuda_runtime.h>

#include <cstdio>
#include <iostream>

// #include "cuda_prefix_sum/internal/device_helpers.cuh"
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
    RowMajorKernelArrayView shared_array{
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
      RowMajorKernelArrayView right_edges_buffer,
      RowMajorKernelArrayView bottom_edges_buffer
  ) {
    extern __shared__ int shared_mem[];

    RowMajorKernelArrayView shared_array{
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

  __global__ void ApplyTileGlobalOffsets(
      KernelLaunchParams params,
      RowMajorKernelArrayViewConst right_edge_prefixes,
      RowMajorKernelArrayViewConst bottom_edge_prefixes
  ) {
    ArraySize2D sub_tile_size = params.sub_tile_size;
    RowMajorKernelArrayView global_array = params.array;

    for (int local_row = 0; local_row < sub_tile_size.num_rows; ++local_row) {
      for (int local_col = 0; local_col < sub_tile_size.num_cols;
           ++local_col) {
        // Compute coordinates
        int global_row = blockIdx.y * blockDim.y * sub_tile_size.num_rows +
                         threadIdx.y * sub_tile_size.num_rows + local_row;
        int global_col = blockIdx.x * blockDim.x * sub_tile_size.num_cols +
                         threadIdx.x * sub_tile_size.num_cols + local_col;

        int tile_row = blockIdx.y;
        int tile_col = blockIdx.x;

        if (global_row >= global_array.size.num_rows ||
            global_col >= global_array.size.num_cols)
          return;

        int adjustment = 0;
        if (tile_col > 0) {
          adjustment += right_edge_prefixes.At(global_row, tile_col);
        }
        if (tile_row > 0) {
          adjustment += bottom_edge_prefixes.At(tile_row, global_col);
        }
        // if (tile_row > 0 && tile_col > 0) {
        //   // Pick the corner value once — it's stored at the bottom of the
        //   // above tile (in bottom_edge_prefixes) OR at the right of the left
        //   // tile
        //   int corner_value = bottom_edge_prefixes.At(
        //       tile_row,
        //       sub_tile_size.num_cols * (tile_col - 1) + sub_tile_size.num_cols + 1
        //   );
        //   adjustment += corner_value;
        // }

        global_array.At(global_row, global_col) += adjustment;
      }
    }
  }

} // namespace subtile_kernels
