#pragma once

#include "cuda_prefix_sum/internal/kernel_launch_params.hpp"
#include "cuda_prefix_sum/kernel_launcher.hpp"

class MultiTileKernelLauncher : public KernelLauncher {
public:
  MultiTileKernelLauncher(const ProgramArgs& program_args);
  ~MultiTileKernelLauncher();
  void Launch() override;

private:
  int* device_array_ = nullptr;
  int* tile_right_edges_buffer_ = nullptr;
  int* tile_bottom_edges_buffer_ = nullptr;
  KernelLaunchParams launch_params_;
  const ProgramArgs &program_args_;
  void AllocateDeviceMemory();
  void FreeDeviceMemory();
  dim3 GetBlockDim();
  dim3 GetGridDim();
  size_t GetSharedMemSize();
  void CheckErrors();
};
