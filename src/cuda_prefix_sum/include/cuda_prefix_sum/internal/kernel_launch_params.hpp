#pragma once

#include "common/program_args.hpp"

// Non-member operator==
inline bool operator==(const ArraySize2D& lhs, const ArraySize2D& rhs) {
  return lhs.num_rows == rhs.num_rows && lhs.num_cols == rhs.num_cols;
}

inline bool operator!=(const ArraySize2D& lhs, const ArraySize2D& rhs) {
  return !(lhs == rhs);
}


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
