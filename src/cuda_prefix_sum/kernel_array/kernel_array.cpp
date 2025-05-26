
#include "cuda_prefix_sum/internal/kernel_array.hpp"

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/array_size_2d.hpp"

RowMajorKernelArray::RowMajorKernelArray(ArraySize2D size)
    : size_{size} {
  size_t bytes = size.num_rows * size.num_cols * sizeof(int);
  cudaError_t err = cudaMalloc(&d_address_, bytes);
  if (err != cudaSuccess) {
    throw std::runtime_error(
        "cudaMalloc failed: " + std::string(cudaGetErrorString(err))
    );
  }
}

RowMajorKernelArray::~RowMajorKernelArray() {
  if (d_address_) {
    cudaFree(d_address_);
    d_address_ = nullptr;
  }
}

RowMajorKernelArrayView RowMajorKernelArray::View() const {
  return RowMajorKernelArrayView{d_address_, size_};
}

RowMajorKernelArrayViewConst RowMajorKernelArray::ConstView() const {
  return RowMajorKernelArrayViewConst{d_address_, size_};
}

int *RowMajorKernelArray::d_address() { return d_address_; }

void RowMajorKernelArray::DebugPrintOnHost(const std::string &label) const {
  std::vector<int> host_data(size_.num_rows * size_.num_cols);
  cudaMemcpy(
      host_data.data(),
      d_address_,
      host_data.size() * sizeof(int),
      cudaMemcpyDeviceToHost
  );

  std::cout << label << ":" << std::endl;
  for (size_t row = 0; row < size_.num_rows; ++row) {
    for (size_t col = 0; col < size_.num_cols; ++col) {
      std::cout << host_data[row * size_.num_cols + col] << "\t";
    }
    std::cout << std::endl;
  }
}