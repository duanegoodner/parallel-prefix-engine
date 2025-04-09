#include "cuda_prefix_sum/kernel_launch_params.hpp"

#include "common/program_args.hpp"

KernelLaunchParams CreateKernelLaunchParams(const ProgramArgs &args) {
  return KernelLaunchParams{
      .full_matrix_dim_x = args.full_matrix_dim()[0],
      .full_matrix_dim_y = args.full_matrix_dim()[1],
      .tile_size_x = args.tile_dim()[0],
      .tile_size_y = args.tile_dim()[1],
  };
}