#pragma once

#include "cuda_prefix_sum/internal/kernel_launch_params.hpp"
#include "cuda_prefix_sum/internal/kernel_launcher.hpp"

class MultiBlockKernelLauncher : public KernelLauncher {
public:
  void Launch(const KernelLaunchParams &params) override;
};
