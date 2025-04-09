#include "cuda_prefix_sum/kernel_launch_params.hpp"

#include "common/program_args.hpp"

KernelLaunchParams CreateKernelLaunchParams(int* d_arr, const ProgramArgs &args) {
  KernelArray kernel_array{
    .d_address = d_arr,
    .size = ArraySize2D(args.full_matrix_dim()[0],args.full_matrix_dim()[1])
    // .size_x = args.full_matrix_dim()[0],
    // .size_y = args.full_matrix_dim()[1]
  };
  
  return KernelLaunchParams{
      .array = kernel_array,
      // .arr_size_x = args.full_matrix_dim()[0],
      // .arr_size_y = args.full_matrix_dim()[1],
      .tile_size = ArraySize2D(args.tile_dim()[0],args.tile_dim()[1])
      // .tile_size_x = args.tile_dim()[0],
      // .tile_size_y = args.tile_dim()[1],
  };
}

