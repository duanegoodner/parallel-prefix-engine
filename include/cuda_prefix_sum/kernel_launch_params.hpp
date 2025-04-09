#pragma once

#include "common/program_args.hpp"

struct ArraySize2D {
  int x;
  int y;
};

struct KernelArray {
  int *d_address;
  ArraySize2D size;
};

struct KernelLaunchParams {
  KernelArray array;
  ArraySize2D tile_size;
};

KernelLaunchParams CreateKernelLaunchParams(
    int *d_arr,
    const ProgramArgs &program_args
);

