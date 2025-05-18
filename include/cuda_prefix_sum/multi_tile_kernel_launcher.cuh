#pragma once

#include "cuda_prefix_sum/internal/kernel_launch_params.hpp"
#include "cuda_prefix_sum/kernel_launcher.hpp"

class MultiTileKernelLauncher : public KernelLauncher {
public:
  MultiTileKernelLauncher(const ProgramArgs &program_args);
  ~MultiTileKernelLauncher();
  void Launch(int* data_array) override;

private:
  const ProgramArgs &program_args_;
  int* right_tile_edge_buffers_ = nullptr;
  int* bottom_tile_edge_buffers_ = nullptr;
  void AllocateTileEdgeBuffers();
  void FreeTileEdgeBuffers();
  dim3 GetBlockDim();
  dim3 GetGridDim();
  size_t GetSharedMemPerBlock();
  void CheckErrors();
};
