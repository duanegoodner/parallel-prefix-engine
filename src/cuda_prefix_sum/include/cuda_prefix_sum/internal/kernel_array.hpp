#pragma once

#include <cuda_runtime.h>

#include <string>

#include "common/array_size_2d.hpp"
#include "cuda_prefix_sum/internal/kernel_array_view.cuh"

class KernelArray {
public:
  KernelArray(ArraySize2D size);
  ~KernelArray();
  KernelArrayView View() const;
  KernelArrayViewConst ViewConst() const;
  int *d_address();
  ArraySize2D size() {return size_; }
  void DebugPrintOnHost(const std::string& label);

private:
  int *d_address_ = nullptr;
  ArraySize2D size_;
};
