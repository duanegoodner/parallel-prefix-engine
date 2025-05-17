#pragma once
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <stdexcept>

#include "cuda_prefix_sum/internal/kernel_launch_params.hpp"
#include "cuda_prefix_sum/kernel_launcher.hpp"

class SingleTileKernelLauncher : public KernelLauncher {
public:
  SingleTileKernelLauncher(const ProgramArgs& program_args);
  ~SingleTileKernelLauncher();
  void Launch() override;

private:
  int *device_array_ = nullptr;
  KernelLaunchParams launch_params_;
  const ProgramArgs &program_args_;
  void AllocateDeviceMemory();
  void FreeDeviceMemory();
  dim3 GetBlockDim();
  dim3 GetGridDim();
  size_t GetSharedMemSize();
  void CheckErrors();
  void CheckProvidedTileSize();
};
