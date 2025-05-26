#pragma once

#include "common/array_size_2d.hpp"
#include "common/program_args.hpp"
#include "cuda_prefix_sum/internal/kernel_array.hpp"



struct KernelLaunchParams {
  RowMajorKernelArrayView array;
  ArraySize2D tile_size;
  ArraySize2D sub_tile_size;
};

KernelLaunchParams CreateKernelLaunchParams(
    const RowMajorKernelArray &device_array,
    const ProgramArgs &program_args
);
