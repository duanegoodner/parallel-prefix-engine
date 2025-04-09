#pragma once

#include "common/program_args.hpp"

struct KernelLaunchParams {
  int* d_arr;
  int arr_size_x;
  int arr_size_y;
  int tile_size_x;
  int tile_size_y;
};

KernelLaunchParams CreateKernelLaunchParams(int* d_arr, const ProgramArgs &program_args);