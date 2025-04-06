// cuda_prefix_sum_solver.cu
//
// Defines the CUDA kernel and launch function for performing 2D prefix sum.
// This file contains only GPU-side logic and is compiled by NVCC.

#include <cuda_runtime.h>

#include <cstdio>
#include <iostream>

#include "common/program_args.hpp"

#include "cuda_prefix_sum/cuda_prefix_sum_solver.cuh" // Host-side declaration

__device__ void PrintSharedMemoryArray(
    const int *array,
    int rows_per_tile,
    int cols_per_tile,
    const char *label
) {
  // Only thread (0, 0) in the block prints the array to avoid excessive output
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    printf("%s:\n", label);
    for (int i = 0; i < rows_per_tile; ++i) {
      for (int j = 0; j < cols_per_tile; ++j) {
        printf("%d ", array[i * rows_per_tile + j]);
      }
      printf("\n");
    }
  }
  __syncthreads(); // Ensure all threads reach this point
}

__global__ void PrefixSumKernel(
    int *d_data,
    int rows_per_tile,
    int cols_per_tile
) {
  // Declare dynamic shared memory
  extern __shared__ int shared_mem[];

  // Divide shared memory into two arrays
  int *arrayA = shared_mem;
  int *arrayB = shared_mem + rows_per_tile * cols_per_tile;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int index = ty * rows_per_tile + tx;

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
  arrayA[ty * rows_per_tile + tx] = d_data[index];
  __syncthreads();

  // Debug statement: Print contents of arrayA after loading from global memory
  PrintSharedMemoryArray(
      arrayA,
      rows_per_tile,
      cols_per_tile,
      "Contents of arrayA after loading from global memory"
  );

  // === Phase 2: Row-wise prefix sum into arrayB ===
  int sum = 0;
  for (int i = 0; i <= tx; ++i) {
    sum += arrayA[ty * rows_per_tile + i];
  }
  arrayB[ty * rows_per_tile + tx] = sum;

  __syncthreads();

  // Debug: Print contents of arrayB after row-wise prefix sum
  PrintSharedMemoryArray(
      arrayB,
      rows_per_tile,
      cols_per_tile,
      "Contents of arrayB after row-wise prefix sum"
  );

  // === Phase 3: Column-wise prefix sum (over arrayB) into arrayA ===
  sum = 0;
  for (int i = 0; i <= ty; ++i) {
    sum += arrayB[i * cols_per_tile + tx];
  }
  arrayA[ty * rows_per_tile + tx] = sum;

  __syncthreads();

  // Debug: Print contents of arrayA after column-wise prefix sum
  PrintSharedMemoryArray(
      arrayA,
      rows_per_tile,
      cols_per_tile,
      "Contents of arrayA after column-wise prefix sum"
  );

  // === Phase 4: Write final result back to global memory ===
  d_data[index] = arrayA[ty * rows_per_tile + tx];
}

void LaunchPrefixSumKernel(
    int *d_data,
    int rows_per_block,
    int cols_per_block,
    int blocks_per_row,
    int blocks_per_col,
    cudaStream_t stream
) {
  std::cout << "Launching PrefixSumKernel with dimensions: " << blocks_per_row
            << "x" << blocks_per_col << " and block size: " << rows_per_block
            << "x" << cols_per_block << std::endl;
  dim3 blockDim(rows_per_block, cols_per_block);
  dim3 gridDim(1, 1); // Single block for now

  PrefixSumKernel<<<gridDim, blockDim, 0, stream>>>(
      d_data,
      rows_per_block,
      cols_per_block
  );

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }

  cudaDeviceSynchronize();
}

__device__ void PrintSharedMemoryArrayNew(
    const int *array,
    int num_rows,
    int num_cols
) {

  printf("Calling PrintharedMemoryArrayNew");
  // Only thread (0, 0) in the block prints the array to avoid excessive output
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    for (int row = 0; row < num_rows; ++row) {
      for (int col = 0; col < num_cols; ++col) {
        printf("%d ", array[row * num_cols + col]);
      }
      printf("\n");
    }
  }
  __syncthreads(); // Ensure all threads reach this point
}

__global__ void PrefixSumKernelNew(
    int *d_data,
    const KernelParams &kernel_params
) {
  // Declare dynamic shared memory
  extern __shared__ int shared_mem[];

  // Divide shared memory into two arrays
  int *arrayA = shared_mem;
  int *arrayB = shared_mem + kernel_params.full_matrix_dim_x *
                                 kernel_params.full_matrix_dim_y;
  ;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int thread_index_1d = tx * kernel_params.num_tile_cols + ty;

  // Debug statement: Print thread and block indices
  printf(
      "Block (%d, %d), Thread (%d, %d), Global Index: %d\n",
      blockIdx.x,
      blockIdx.y,
      threadIdx.x,
      threadIdx.y,
      thread_index_1d
  );

  // === Phase 1: Load input from global memory to shared memory ===
  for (auto in_tile_row = 0; in_tile_row < kernel_params.tile_dim_x;
       ++in_tile_row) {
    for (auto in_tile_col = 0; in_tile_col < kernel_params.tile_dim_y;
         ++in_tile_col) {
      int array_x = threadIdx.x * kernel_params.tile_dim_x + in_tile_row;
      int array_y = threadIdx.y * kernel_params.tile_dim_y + in_tile_col;
      int idx_1d = array_x * kernel_params.full_matrix_dim_x *
                       kernel_params.full_matrix_dim_y +
                   array_y;
      arrayA[idx_1d] = d_data[idx_1d];
    }
  }

  __syncthreads();

  // Debug statement: Print contents of arrayA after loading from global memory
  PrintSharedMemoryArrayNew(
      arrayA,
      kernel_params.full_matrix_dim_x,
      kernel_params.full_matrix_dim_y
  );

  // // === Phase 2: Row-wise prefix sum into arrayB ===
  // int sum = 0;
  // for (int i = 0; i <= tx; ++i) {
  //   sum += arrayA[ty * rows_per_tile + i];
  // }
  // arrayB[ty * rows_per_tile + tx] = sum;

  // __syncthreads();

  // // Debug: Print contents of arrayB after row-wise prefix sum
  // PrintSharedMemoryArray(
  //     arrayB,
  //     rows_per_tile,
  //     cols_per_tile,
  //     "Contents of arrayB after row-wise prefix sum"
  // );

  // // === Phase 3: Column-wise prefix sum (over arrayB) into arrayA ===
  // sum = 0;
  // for (int i = 0; i <= ty; ++i) {
  //   sum += arrayB[i * cols_per_tile + tx];
  // }
  // arrayA[ty * rows_per_tile + tx] = sum;

  // __syncthreads();

  // // Debug: Print contents of arrayA after column-wise prefix sum
  // PrintSharedMemoryArray(
  //     arrayA,
  //     rows_per_tile,
  //     cols_per_tile,
  //     "Contents of arrayA after column-wise prefix sum"
  // );

  // === Phase 4: Write final result back to global memory ===
  // d_data[index] = arrayA[ty * rows_per_tile + tx];
}

void LaunchPrefixSumKernelNew(
    int *d_data,
    const ProgramArgs &program_args,
    cudaStream_t stream
) {

  KernelParams kernel_params{program_args};

  std::cout << "LaunchPrefixSumKernelNew called with: " << std::endl;
  kernel_params.Print();
  dim3 block_dim(kernel_params.num_tile_rows,kernel_params.num_tile_cols);
  dim3 grid_dim(1, 1);

  PrefixSumKernelNew<<<grid_dim, block_dim, 0, stream>>>(
    d_data,
    kernel_params
);
}