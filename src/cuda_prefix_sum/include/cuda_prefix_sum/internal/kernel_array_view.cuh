#pragma once

#include <cuda_runtime.h>

#include "common/array_size_2d.hpp"

struct KernelArrayView {
  int *d_address;
  ArraySize2D size;

  __device__ __host__ int Index1D(int row, int col) const {
    return row * size.num_cols + col;
  }

  __device__ __host__ const int &At(int row, int col) const {
    return d_address[Index1D(row, col)];
  }

  __device__ __host__ int &At(int row, int col) {
    return d_address[Index1D(row, col)];
  }
};

struct KernelArrayViewConst {
  const int *d_address;
  ArraySize2D size;

  __device__ __host__ int Index1D(int row, int col) const {
    return row * size.num_cols + col;
  }

  __device__ __host__ const int &At(int row, int col) const {
    return d_address[Index1D(row, col)];
  }
};
