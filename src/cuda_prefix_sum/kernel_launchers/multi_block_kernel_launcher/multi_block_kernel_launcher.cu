#include "cuda_prefix_sum/multi_block_kernel_launcher.cuh"

#include <cuda_runtime.h>
#include <iostream>

#include "cuda_prefix_sum/internal/kernel_launch_params.hpp"
#include "cuda_prefix_sum/internal/multi_block_kernel.cuh"

void MultiBlockKernelLauncher::Launch(const KernelLaunchParams &params) {
  const ArraySize2D &full_size = params.array.size;
  const ArraySize2D &tile_size = params.tile_size;
  const ArraySize2D &subtile_size = params.sub_tile_size;

  // Threads per block
  dim3 blockDim(
      tile_size.num_cols / subtile_size.num_cols,
      tile_size.num_rows / subtile_size.num_rows
  );

  // Blocks per grid
  dim3 gridDim(
      full_size.num_cols / tile_size.num_cols,
      full_size.num_rows / tile_size.num_rows
  );

  int sharedMemBytes = tile_size.num_rows * tile_size.num_cols * sizeof(int);

  std::cout << "[CUDA] Launching MultiBlockKernel with gridDim = ("
            << gridDim.x << ", " << gridDim.y
            << "), blockDim = (" << blockDim.x << ", " << blockDim.y
            << "), sharedMem = " << sharedMemBytes << " bytes" << std::endl;

  MultiBlockKernel<<<gridDim, blockDim, sharedMemBytes>>>(params);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("Kernel launch failed: ") +
                             cudaGetErrorString(err));
  }
}
