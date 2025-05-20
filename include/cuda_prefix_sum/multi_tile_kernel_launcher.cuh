#pragma once

#include "cuda_prefix_sum/internal/kernel_array.hpp"
#include "cuda_prefix_sum/internal/kernel_launch_params.hpp"
#include "cuda_prefix_sum/kernel_launcher.hpp"

class MultiTileKernelLauncher : public KernelLauncher {
public:
  MultiTileKernelLauncher(const ProgramArgs &program_args);
  void Launch(const KernelArray &device_array) override;

private:
  const ProgramArgs &program_args_;
  KernelArray right_tile_edge_buffers_;
  KernelArray bottom_tile_edge_buffers_;
  // int* right_tile_edge_buffers_ = nullptr;
  // int* bottom_tile_edge_buffers_ = nullptr;
  dim3 GetBlockDim();
  dim3 GetGridDim();
  size_t GetSharedMemPerBlock();
  void CheckErrors();
};
