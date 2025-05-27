#pragma once

#include <cuda_runtime.h>

#include <string>

#include "common/array_size_2d.hpp"
#include "cuda_prefix_sum/internal/kernel_array_view.cuh"

class RowMajorKernelArray {
public:
  RowMajorKernelArray(ArraySize2D size);
  ~RowMajorKernelArray();
  int At(size_t row, size_t col);
  RowMajorKernelArrayView View() const;
  RowMajorKernelArrayViewConst ConstView() const;
  int *d_address();
  ArraySize2D size() {return size_; }
  void DebugPrintOnHost(const std::string& label) const;

private:
  size_t Index1D(size_t row, size_t col);
  int *d_address_ = nullptr;
  ArraySize2D size_;
};
