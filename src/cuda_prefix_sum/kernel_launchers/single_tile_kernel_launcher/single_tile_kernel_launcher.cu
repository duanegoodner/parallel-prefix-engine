#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <string>

#include "common/logger.hpp"
#include "common/program_args.hpp"

#include "cuda_prefix_sum/internal/kernel_config_utils.cuh"
#include "cuda_prefix_sum/internal/kernel_launch_params.hpp"
#include "cuda_prefix_sum/internal/single_tile_kernel.cuh"
#include "cuda_prefix_sum/single_tile_kernel_launcher.cuh"

SingleTileKernelLauncher::SingleTileKernelLauncher(
    const ProgramArgs &program_args
)
    : program_args_{program_args} {
  AllocateDeviceMemory();
  launch_params_ = CreateKernelLaunchParams(device_array_, program_args_);
}

SingleTileKernelLauncher::~SingleTileKernelLauncher() { FreeDeviceMemory(); }

void SingleTileKernelLauncher::AllocateDeviceMemory() {
  cudaMalloc(&device_array_, program_args_.FullMatrixSize() * sizeof(int));
}

void SingleTileKernelLauncher::FreeDeviceMemory() {
  if (device_array_) {
    cudaFree(device_array_);
    device_array_ = nullptr;
  }
}

void SingleTileKernelLauncher::Launch() {

  CheckProvidedTileSize();

  // Set max dynamic shared memory and prefer shared over L1
  constexpr size_t kMaxSharedMemBytes = 98304;
  ConfigureSharedMemoryForKernel(SingleTileKernel, kMaxSharedMemBytes);

  // Prepare launch configuration
  dim3 block_dim = GetBlockDim();
  dim3 grid_dim = GetGridDim();
  size_t shared_mem_size = GetSharedMemSize();

  // Launch the kernel
  SingleTileKernel<<<grid_dim, block_dim, shared_mem_size, 0>>>(launch_params_
  );

  // Validate
  CheckErrors();
}

dim3 SingleTileKernelLauncher::GetGridDim() { return dim3(1, 1, 1); }

dim3 SingleTileKernelLauncher::GetBlockDim() {
  if (launch_params_.sub_tile_size.num_rows == 0 ||
      launch_params_.sub_tile_size.num_cols == 0) {
    throw std::invalid_argument("Sub-tile size dimensions must be non-zero");
  }

  uint32_t num_tile_rows = static_cast<uint32_t>(
      launch_params_.array.size.num_rows /
      launch_params_.sub_tile_size.num_rows
  );
  uint32_t num_tile_cols = static_cast<uint32_t>(
      launch_params_.array.size.num_cols /
      launch_params_.sub_tile_size.num_cols
  );

  return dim3(num_tile_cols, num_tile_rows, 1);
}

size_t SingleTileKernelLauncher::GetSharedMemSize() {
  return static_cast<size_t>(launch_params_.array.size.num_rows) *
         static_cast<size_t>(launch_params_.array.size.num_cols) * sizeof(int);
}

void SingleTileKernelLauncher::CheckErrors() {
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

void SingleTileKernelLauncher::CheckProvidedTileSize() {
  if (launch_params_.array.size != launch_params_.tile_size) {
    std::cout << std::endl;
    std::string tile_size = std::to_string(launch_params_.tile_size.num_rows) +
                            "x" +
                            std::to_string(launch_params_.tile_size.num_cols);
    std::string full_matrix_size =
        std::to_string(launch_params_.array.size.num_rows) + "x" +
        std::to_string(launch_params_.array.size.num_cols) + ".";
    Logger::Log(
        LogLevel::WARNING,
        "Specified tile size of " + tile_size +
            " does not match full matrix size of " + full_matrix_size
    );
    Logger::Log(
        LogLevel::WARNING,
        "Single tile kernel uses single top level tile with size equal to "
        "full matrix size."
    );
    Logger::Log(
        LogLevel::WARNING,
        "Ignoring provided tile size value of " + tile_size + "."
    );
    std::cout << std::endl;
  }
}