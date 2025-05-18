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
  AllocateTileEdgeBuffers();
}

MultiTileKernelLauncher::~MultiTileKernelLauncher() { FreeTileEdgeBuffers(); }

void MultiTileKernelLauncher::AllocateTileEdgeBuffers() {
  cudaMalloc(
      &right_tile_edge_buffers_,
      program_args_.FullMatrixSize2D().num_rows * GetGridDim().x * sizeof(int)
  );
  cudaMalloc(
      &bottom_tile_edge_buffers_,
      program_args_.FullMatrixSize2D().num_cols * GetGridDim().y * sizeof(int)
  );
}

void MultiTileKernelLauncher::FreeTileEdgeBuffers() {
  if (right_tile_edge_buffers_) {
    cudaFree(right_tile_edge_buffers_);
    right_tile_edge_buffers_ = nullptr;
  }
  if (bottom_tile_edge_buffers_) {
    cudaFree(bottom_tile_edge_buffers_);
    bottom_tile_edge_buffers_ = nullptr;
  }
}

void MultiTileKernelLauncher::Launch(int *data_array) {
  constexpr size_t kMaxSharedMemBytes = 98304;
  ConfigureSharedMemoryForKernel(MultiTileKernel, kMaxSharedMemBytes);

  // Prepare launch configuration
  dim3 block_dim = GetBlockDim();
  dim3 grid_dim = GetGridDim();
  size_t shared_mem_size = GetSharedMemPerBlock();

  auto launch_params = CreateKernelLaunchParams(data_array, program_args_);

  MultiTileKernel<<<grid_dim, block_dim, shared_mem_size>>>(
      launch_params,
      right_tile_edge_buffers_,
      bottom_tile_edge_buffers_
  );

  CheckErrors();

  



}

dim3 MultiTileKernelLauncher::GetGridDim() {

  auto num_block_rows = program_args_.FullMatrixSize2D().num_rows /
                        program_args_.TileSize2D().num_rows;

  auto num_block_cols = program_args_.FullMatrixSize2D().num_cols /
                        program_args_.TileSize2D().num_cols;

  return dim3(num_block_cols, num_block_rows, 1);
}

dim3 MultiTileKernelLauncher::GetBlockDim() {
  auto num_thread_rows = program_args_.TileSize2D().num_rows /
                         program_args_.SubTileSize2D().num_rows;

  auto num_thread_cols = program_args_.TileSize2D().num_cols /
                         program_args_.SubTileSize2D().num_cols;

  return dim3(num_thread_cols, num_thread_rows, 1);
}

size_t MultiTileKernelLauncher::GetSharedMemPerBlock() {

  return static_cast<size_t>(program_args_.TileSize2D().num_rows) *
         static_cast<size_t>(program_args_.TileSize2D().num_cols) *
         sizeof(int);
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
