#pragma once

// #include <cuda_runtime.h>

#include "common/array_size_2d.hpp"

struct KernelArrayView {
  int *d_address;
  ArraySize2D size;
};

struct KernelArray {
  int *d_address;
  ArraySize2D size;

//   KernelArray(ArraySize2D size);
//   KernelArrayView View() const;
};
