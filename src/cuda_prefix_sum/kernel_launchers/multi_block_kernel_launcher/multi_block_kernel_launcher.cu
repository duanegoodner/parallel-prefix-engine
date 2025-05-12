#include <cuda_runtime.h>

#include <iostream>

#include "cuda_prefix_sum/internal/kernel_config_utils.cuh"
#include "cuda_prefix_sum/internal/kernel_launch_params.hpp"
#include "cuda_prefix_sum/internal/multi_block_kernel.cuh"
#include "cuda_prefix_sum/multi_block_kernel_launcher.cuh"

void MultiBlockKernelLauncher::Launch(
    const KernelLaunchParams &launch_params
) {
  constexpr size_t kMaxSharedMemBytes = 98304;
  ConfigureSharedMemoryForKernel(MultiBlockKernel, kMaxSharedMemBytes);

  // Prepare launch configuration
  dim3 block_dim = GetBlockDim(launch_params);
  dim3 grid_dim = GetGridDim(launch_params);
  size_t shared_mem_size = GetSharedMemPerBlock(launch_params);

  MultiBlockKernel<<<grid_dim, block_dim, shared_mem_size>>>(launch_params);

  CheckErrors();
}

dim3 MultiBlockKernelLauncher::GetGridDim(
    const KernelLaunchParams &launch_params
) {
  auto num_block_rows =
      launch_params.array.size.num_rows / launch_params.tile_size.num_rows;

  auto num_block_cols =
      launch_params.array.size.num_cols / launch_params.tile_size.num_cols;

  return dim3(num_block_cols, num_block_rows, 1);
}

dim3 MultiBlockKernelLauncher::GetBlockDim(
    const KernelLaunchParams &launch_params
) {
  auto num_thread_rows =
      launch_params.tile_size.num_rows / launch_params.sub_tile_size.num_rows;

  auto num_thread_cols =
      launch_params.tile_size.num_cols / launch_params.sub_tile_size.num_cols;

  return dim3(num_thread_cols, num_thread_rows, 1);
}

size_t MultiBlockKernelLauncher::GetSharedMemPerBlock(
    const KernelLaunchParams &launch_params
) {
  return static_cast<size_t>(launch_params.array.size.num_rows) *
         static_cast<size_t>(launch_params.array.size.num_cols) * sizeof(int);
}

void MultiBlockKernelLauncher::CheckErrors() {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }

  cudaDeviceSynchronize();
  cudaError_t sync_err = cudaGetLastError();
  if (sync_err != cudaSuccess) {
    fprintf(stderr, "CUDA sync error: %s\n", cudaGetErrorString(sync_err));
  }
}
