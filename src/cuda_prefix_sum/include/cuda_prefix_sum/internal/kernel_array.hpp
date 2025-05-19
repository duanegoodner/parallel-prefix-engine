#pragma once

// #include <cuda_runtime.h>

#include "common/array_size_2d.hpp"

struct KernelArrayView {
  int *d_address;
  ArraySize2D size;
};

class KernelArray {
public:
  KernelArray(ArraySize2D size);
  ~KernelArray();
  KernelArrayView View() const;

private:
  int *d_address_ = nullptr;
  ArraySize2D size_;
};
