#include <cuda_runtime.h>

#include <iostream>

#include "cuda_prefix_sum/internal/kernel_config_utils.cuh"
#include "cuda_prefix_sum/internal/kernel_launch_params.hpp"
#include "cuda_prefix_sum/internal/multi_tile_kernel.cuh"
#include "cuda_prefix_sum/multi_tile_kernel_launcher.cuh"

MultiTileKernelLauncher::MultiTileKernelLauncher(
    const ProgramArgs &program_args
)
    : program_args_{program_args} {
  AllocateDeviceMemory();
  launch_params_ = CreateKernelLaunchParams(device_array_, program_args_);
}

MultiTileKernelLauncher::~MultiTileKernelLauncher() { FreeDeviceMemory(); }

void MultiTileKernelLauncher::AllocateDeviceMemory() {
  cudaMalloc(&device_array_, program_args_.FullMatrixSize() * sizeof(int));
  cudaMalloc(
      &tile_right_edges_buffer_,
      launch_params_.array.size.num_rows * GetGridDim().x * sizeof(int)
  );
  cudaMalloc(
      &tile_bottom_edges_buffer_,
      launch_params_.array.size.num_cols * GetGridDim().y * sizeof(int)
  );
}

void MultiTileKernelLauncher::FreeDeviceMemory() {
  if (device_array_) {
    cudaFree(device_array_);
    device_array_ = nullptr;
  }

  if (tile_right_edges_buffer_) {
    cudaFree(tile_right_edges_buffer_);
    tile_right_edges_buffer_ = nullptr;
  }

  if (tile_bottom_edges_buffer_) {
    cudaFree(tile_bottom_edges_buffer_);
    tile_bottom_edges_buffer_ = nullptr;
  }
}

void MultiTileKernelLauncher::Launch() {
  constexpr size_t kMaxSharedMemBytes = 98304;
  ConfigureSharedMemoryForKernel(MultiTileKernel, kMaxSharedMemBytes);

  // Prepare launch configuration
  dim3 block_dim = GetBlockDim();
  dim3 grid_dim = GetGridDim();
  size_t shared_mem_size = GetSharedMemSize();

  MultiTileKernel<<<grid_dim, block_dim, shared_mem_size>>>(launch_params_);

  CheckErrors();
}

dim3 MultiTileKernelLauncher::GetGridDim() {
  auto num_block_rows =
      launch_params_.array.size.num_rows / launch_params_.tile_size.num_rows;

  auto num_block_cols =
      launch_params_.array.size.num_cols / launch_params_.tile_size.num_cols;

  return dim3(num_block_cols, num_block_rows, 1);
}

dim3 MultiTileKernelLauncher::GetBlockDim() {
  auto num_thread_rows = launch_params_.tile_size.num_rows /
                         launch_params_.sub_tile_size.num_rows;

  auto num_thread_cols = launch_params_.tile_size.num_cols /
                         launch_params_.sub_tile_size.num_cols;

  return dim3(num_thread_cols, num_thread_rows, 1);
}

size_t MultiTileKernelLauncher::GetSharedMemSize() {
  return static_cast<size_t>(launch_params_.array.size.num_rows) *
         static_cast<size_t>(launch_params_.array.size.num_cols) * sizeof(int);
}

void MultiTileKernelLauncher::CheckErrors() {
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
