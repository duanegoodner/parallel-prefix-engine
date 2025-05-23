#pragma once

#include "common/array_size_2d.hpp"

#include "cuda_prefix_sum/internal/kernel_array_view.cuh"
#include "cuda_prefix_sum/internal/kernel_launch_params.hpp"

namespace device_helpers {

  namespace blockrow {

    namespace chunks {
      

    } // namespace chunks
  } // namespace blockrow

  namespace blockcol {
    __forceinline__ __device__ int RowIndex() { return threadIdx.x; }
    __forceinline__ __device__ int ColIndex() { return blockIdx.x; }
    namespace chunks {}
  } // namespace blockcol
} // namespace device_helpers
