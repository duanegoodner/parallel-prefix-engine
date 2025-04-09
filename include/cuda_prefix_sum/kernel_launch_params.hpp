#pragma once

#include "common/program_args.hpp"

struct KernelLaunchParams {
  int full_matrix_dim_x;
  int full_matrix_dim_y;
  int tile_dim_x;
  int tile_dim_y;
};

KernelLaunchParams CreateKernelLaunchParams(const ProgramArgs &program_args);