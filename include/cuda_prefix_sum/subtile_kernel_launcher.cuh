#pragma once
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <stdexcept>

#include "cuda_prefix_sum/internal/kernel_launch_params.hpp"
#include "cuda_prefix_sum/internal/kernel_launcher.hpp"

class SubTileKernelLauncher : public KernelLauncher {
public:
  void Launch(const KernelLaunchParams &launch_params) override;

private:
  dim3 GetBlockDim(const KernelLaunchParams &launch_params);
  dim3 GetGridDim(const KernelLaunchParams &launch_params);
  size_t GetSharedMemSize(const KernelLaunchParams &launch_params);
  void CheckErrors();
  void CheckProvidedTileSize(const KernelLaunchParams &launch_params);
};
