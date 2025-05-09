#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <cstdio>

#include "cuda_prefix_sum/internal/kernel_launch_params.hpp"
#include "cuda_prefix_sum/kernel_launcher.hpp"

class SubTileKernelLauncher : public KernelLauncher {
public:
  void Launch(const KernelLaunchParams &launch_params) override;

private:
  dim3 GetBlockDim(const KernelLaunchParams &launch_params) {
    if (launch_params.tile_size.num_rows == 0 || launch_params.tile_size.num_cols == 0) {
      throw std::invalid_argument("Tile size dimensions must be non-zero");
    }

    uint32_t num_tile_rows = static_cast<uint32_t>(
        launch_params.array.size.num_rows / launch_params.tile_size.num_rows);
    uint32_t num_tile_cols = static_cast<uint32_t>(
        launch_params.array.size.num_cols / launch_params.tile_size.num_cols);

    return dim3(num_tile_cols, num_tile_rows, 1);
  }

  dim3 GetGridDim(const KernelLaunchParams &) {
    return dim3(1, 1, 1);
  }

  size_t GetSharedMemSize(const KernelLaunchParams &launch_params) {
    return static_cast<size_t>(launch_params.array.size.num_rows) *
           static_cast<size_t>(launch_params.array.size.num_cols) *
           sizeof(int);
  }

  void CheckErrors() {
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
};
