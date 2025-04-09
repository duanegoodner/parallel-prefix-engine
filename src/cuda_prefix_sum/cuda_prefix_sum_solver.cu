// cuda_prefix_sum_solver.cu
//
// Defines the CUDA kernel and launch function for performing 2D prefix sum.
// This file contains only GPU-side logic and is compiled by NVCC.

#include <cuda_runtime.h>

#include <cstdio>
#include <iostream>

#include "cuda_prefix_sum/cuda_device_helpers.cuh"
#include "cuda_prefix_sum/cuda_prefix_sum_solver.cuh"
#include "cuda_prefix_sum/kernel_launch_params.hpp"

__global__ void PrefixSumKernel(
    // int *d_data,
    KernelLaunchParams params
) {
  // Declare dynamic shared memory
  extern __shared__ int shared_mem[];

  // Divide shared memory into two arrays
  KernelArray array_a{.d_address = shared_mem, .size = params.array.size};
  KernelArray array_b{.d_address = shared_mem, .size = params.array.size};

  // === Phase 1: Load input from global memory to shared memory ===
  CopyGlobalArrayToSharedArray(params.array, array_a, params.tile_size);

  __syncthreads();

  // Debug statement: Print contents of arrayA after loading from global memory
  PrintKernelArray(
      array_a,
      "Contents of array_a after loading from global memory"
  );

  // === Phase 2: Row-wise prefix sum within each tile of arrayA ===
  for (int tile_col = 1; tile_col < params.tile_size.y; tile_col++) {
    for (int tile_row = 0; tile_row < params.tile_size.x; ++tile_row) {
      ComputeRowWisePrefixSum(array_a, params.tile_size, tile_row, tile_col);
    }
  }

  __syncthreads();
  // Debug statement: Print contents of arrayA after row-wise prefix sum
  PrintKernelArray(array_a, "Contents of array_a after row-wise prefix sum");

  // === Phase 3: Column-wise prefix sum within each tile of arrayA ===
  for (int tile_row = 1; tile_row < params.tile_size.x; tile_row++) {
    for (int tile_col = 0; tile_col < params.tile_size.y; ++tile_col) {
      int full_matrix_x = threadIdx.x * params.tile_size.x + tile_row;
      int full_matrix_y = threadIdx.y * params.tile_size.y + tile_col;
      int full_matrix_x_prev = threadIdx.x * params.tile_size.x + tile_row - 1;
      CombineElementInto(
          array_a,
          full_matrix_x_prev,
          full_matrix_y,
          full_matrix_x,
          full_matrix_y
      );
    }
  }

  __syncthreads();
  // Debug statement: Print contents of arrayA after column-wise prefix sum
  PrintKernelArray(
      array_a,
      "Contents of array_a after column-wise prefix sum"
  );

  // === Phase 4: Compute/write final result into arrayB ===

  // Extract right edges of upstream tiles
  for (int upstream_tile_col = 0; upstream_tile_col < threadIdx.y;
       ++upstream_tile_col) {
    int upstream_tile_full_matrix_col_idx =
        upstream_tile_col * params.tile_size.y + params.tile_size.y - 1;
    for (int tile_row = 0; tile_row < params.tile_size.x; ++tile_row) {
      int full_matrix_row_idx = threadIdx.x * params.tile_size.x + tile_row;
      int edge_val = array_a.d_address
                         [full_matrix_row_idx * array_a.size.y +
                          upstream_tile_full_matrix_col_idx];
      for (int tile_col = 0; tile_col < params.tile_size.y; ++tile_col) {
        int full_matrix_x = threadIdx.x * params.tile_size.x + tile_row;
        int full_matrix_y = threadIdx.y * params.tile_size.y + tile_col;
        array_b.d_address[full_matrix_x * array_b.size.y + full_matrix_y] =
            array_a.d_address[full_matrix_x * array_a.size.y + full_matrix_y] +
            edge_val;
      }
    }
  }

  __syncthreads();
  // Debug statement: Print contents of arrayB after adding upstream right
  // edges
  PrintKernelArray(
      array_b,
      "Contents of array_b extracting/adding right edges of upstream tiles"
  );

  // PrintArray(
  //     arrayB,
  //     params.arr_size_x,
  //     params.arr_size_y,
  //     "Contents of arrayB extracting/adding right edges of upstream
  //     tiles"
  // );

  // Extract bottom edges of upstream tiles

  for (int tile_row = 0; tile_row < params.tile_size.x; ++tile_row) {
    for (int tile_col = 0; tile_col < params.tile_size.y; ++tile_col) {
      int full_matrix_x = threadIdx.x * params.tile_size.x + tile_row;
      int full_matrix_y = threadIdx.y * params.tile_size.y + tile_col;
    }
  }

  // === Phase 2: Row-wise prefix sum into arrayB ===
  int sum = 0;
  for (int col = 0; col <= threadIdx.y; ++col) {
    // TODO: implement op
    sum += array_a.d_address[threadIdx.x * blockDim.y + col];
  }

  array_b.d_address[threadIdx.x * blockDim.y + threadIdx.y] = sum;

  __syncthreads();

  // PrintSharedMemoryArray(
  //     arrayB,
  //     "Contents of arrayB after row-wise prefix sum"
  // );

  // === Phase 3: Column-wise prefix sum (over arrayB) into arrayA ===
  sum = 0;
  for (int row = 0; row <= threadIdx.x; ++row) {
    // TODO: implement op
    sum += array_b.d_address[row * blockDim.y + threadIdx.y];
  }
  array_a.d_address[threadIdx.x * blockDim.y + threadIdx.y] = sum;

  __syncthreads();

  // Debug: Print contents of arrayA after column-wise prefix sum
  // PrintSharedMemoryArray(
  //     arrayA,
  //     "Contents of arrayA after column-wise prefix sum"
  // );

  // === Phase 4: Write final result back to global memory ===
  CopySharedArrayToGlobalArray(array_b, params.array, params.tile_size);

  // params.d_arr[index] = arrayA[tx * blockDim.y + ty];

  // PrintGlobalMemArray(d_data);
}

void LaunchPrefixSumKernel(
    // int *d_data,
    KernelLaunchParams kernel_params,
    cudaStream_t stream
) {

  int num_tiles_x = kernel_params.array.size.x / kernel_params.tile_size.x;
  int num_tiles_y = kernel_params.array.size.y / kernel_params.tile_size.y;

  // dim3 blockDim(full_matrix_dim_x, full_matrix_dim_y);
  dim3 blockDim(num_tiles_x, num_tiles_y);
  dim3 gridDim(1, 1); // Single block for now

  PrefixSumKernel<<<gridDim, blockDim, 0, stream>>>(
      // d_data,
      kernel_params
  );

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }

  cudaDeviceSynchronize();
}
