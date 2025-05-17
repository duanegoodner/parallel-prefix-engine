#pragma once

#include "common/program_args.hpp"

struct KernelArray {
  int *d_address;
  ArraySize2D size;
};

struct KernelLaunchParams {
  KernelArray array;
  ArraySize2D tile_size;
  ArraySize2D sub_tile_size;
};

KernelLaunchParams CreateKernelLaunchParams(
    int *d_arr,
    const ProgramArgs &program_args
);
