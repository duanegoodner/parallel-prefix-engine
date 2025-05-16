#pragma once

#include "cuda_prefix_sum/internal/kernel_launch_params.hpp"
#include "cuda_prefix_sum/kernel_launcher.hpp"

class MultiTileKernelLauncher : public KernelLauncher {
public:
  void Launch(const KernelLaunchParams &params) override;

private:
  dim3 GetBlockDim(const KernelLaunchParams &launch_params);
  dim3 GetGridDim(const KernelLaunchParams &launch_params);
  size_t GetSharedMemPerBlock(const KernelLaunchParams &launch_params);
  void CheckErrors();
};
