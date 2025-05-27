#pragma once

#include "cuda_prefix_sum/internal/kernel_array.hpp"
#include "cuda_prefix_sum/internal/kernel_launch_params.hpp"
#include "cuda_prefix_sum/kernel_launcher.hpp"

class MultiTileKernelLauncher : public KernelLauncher {
public:
  MultiTileKernelLauncher(const ProgramArgs &program_args);
  void Launch(const RowMajorKernelArray &device_array) override;

private:
  const ProgramArgs &program_args_;
  RowMajorKernelArray right_tile_edge_buffers_;
  RowMajorKernelArray right_tile_edge_buffers_ps_;
  RowMajorKernelArray bottom_tile_edge_buffers_;
  RowMajorKernelArray bottom_tile_edge_buffers_ps_;
  size_t buffer_sum_method_cutoff_ = 1024;
  size_t mult_block_buffer_sum_chunk_size_ = 512;

  dim3 FirstPassBlockDim();
  dim3 FirstPassGridDim();
  size_t FirstPassSharedMemPerBlock();

  void LaunchRowWisePrefixSum(
      const int *d_input,
      int *d_output,
      ArraySize2D size
  );

  void LaunchRowToColInjection();

  void LaunchColWisePrefixSum(
      const int *d_input,
      int *d_output,
      ArraySize2D size
  );

  void CheckErrors();
};
