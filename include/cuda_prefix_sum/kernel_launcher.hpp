// kernel_launcher.hpp
#pragma once

#include "cuda_prefix_sum/internal/kernel_array.hpp"
#include "cuda_prefix_sum/internal/kernel_launch_params.hpp"

class KernelLauncher {
public:
  virtual ~KernelLauncher() = default;

  virtual void Launch(const KernelArray &device_array) = 0;
};
