#pragma once
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <stdexcept>

#include "cuda_prefix_sum/internal/kernel_array.hpp"
#include "cuda_prefix_sum/internal/kernel_launch_params.hpp"
#include "cuda_prefix_sum/kernel_launcher.hpp"

class SingleTileKernelLauncher : public KernelLauncher {
public:
  SingleTileKernelLauncher(const ProgramArgs &program_args);

  void Launch(const RowMajorKernelArray &device_array) override;

private:
  const ProgramArgs &program_args_;
  dim3 GetBlockDim();
  dim3 GetGridDim();
  size_t GetSharedMemSize();
  void CheckErrors();
  void CheckProvidedTileSize();
};
