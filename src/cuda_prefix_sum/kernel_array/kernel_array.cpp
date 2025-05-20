
#include "cuda_prefix_sum/internal/kernel_array.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

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

int* KernelArray::d_address() {return d_address_; }