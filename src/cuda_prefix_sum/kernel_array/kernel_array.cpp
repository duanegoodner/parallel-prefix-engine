
#include "cuda_prefix_sum/internal/kernel_array.hpp"

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/array_size_2d.hpp"

KernelArray::KernelArray(ArraySize2D size)
    : size_{size} {
  size_t bytes = size.num_rows * size.num_cols * sizeof(int);
  cudaError_t err = cudaMalloc(&d_address_, bytes);
  if (err != cudaSuccess) {
    throw std::runtime_error(
        "cudaMalloc failed: " + std::string(cudaGetErrorString(err))
    );
  }
}

KernelArray::~KernelArray() {
  if (d_address_) {
    cudaFree(d_address_);
    d_address_ = nullptr;
  }
}

KernelArrayView KernelArray::View() const {
  return KernelArrayView{d_address_, size_};
}

int *KernelArray::d_address() { return d_address_; }

void KernelArray::DebugPrintOnHost(const std::string& label) {
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