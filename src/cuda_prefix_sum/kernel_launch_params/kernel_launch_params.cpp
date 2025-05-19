#include "cuda_prefix_sum/internal/kernel_launch_params.hpp"

#include "common/array_size_2d.hpp"
#include "common/program_args.hpp"

#include <stdexcept>  // Required for std::runtime_error

KernelLaunchParams CreateKernelLaunchParams(
    int *d_arr,
    const ProgramArgs &args
) {
  if (!args.sub_tile_dim().has_value()) {
    throw std::runtime_error(
        "sub_tile_dim must be set when constructing KernelLaunchParams (CUDA backend)"
    );
  }

  const auto &subtile = args.sub_tile_dim().value();  // Safe: we've checked
  ArraySize2D tile_size(args.tile_dim()[0], args.tile_dim()[1]);
  ArraySize2D sub_tile_size(subtile[0], subtile[1]);

  KernelArrayView kernel_array{
      .d_address = d_arr,
      .size = ArraySize2D(args.full_matrix_dim()[0], args.full_matrix_dim()[1])
  };

  return KernelLaunchParams{
      .array = kernel_array,
      .tile_size = tile_size,
      .sub_tile_size = sub_tile_size
  };
}

