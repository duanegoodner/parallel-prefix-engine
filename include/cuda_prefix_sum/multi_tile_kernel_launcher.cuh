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
  KernelArray right_tile_edge_buffers_ps_;
  KernelArray bottom_tile_edge_buffers_;
  KernelArray bottom_tile_edge_buffers_ps_;
  dim3 FirstPassBlockDim();
  dim3 FirstPassGridDim();
  size_t FirstPassSharedMemPerBlock();
  void LaunchRowWisePrefixSum(
      const int *d_input,
      int *d_output,
      // int num_rows,
      // int num_cols,
      ArraySize2D size,
      int chunk_size = 512
  );
  void EdgeBufferRowWisePrefixSum();
  void CheckErrors();
};
