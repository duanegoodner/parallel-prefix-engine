#pragma once

#include "common/array_size_2d.hpp"
#include "cuda_prefix_sum/internal/kernel_array_view.cuh"

namespace edge_buffer_row_kernels {

namespace single_block {
__global__ void RowWiseScanSingleBlock(
    const int *__restrict__ in,
    int *__restrict__ out,
    // int num_cols
    ArraySize2D size
);

}

namespace multi_block {
__global__ void RowWiseScanMultiBlockPhase1(
    const int *__restrict__ in,
    int *__restrict__ out,
    int *__restrict__ block_sums,
    ArraySize2D size,
    int chunk_size
);

__global__ void RowWiseScanMultiBlockPhase2(
    int *__restrict__ out,
    const int *__restrict__ scanned_block_sums,
    ArraySize2D size,
    int chunk_size
);
} // namespace multi_block

// Device function to convert an inclusive scan to exclusive for a single row.
// The caller is responsible for ensuring shared_temp contains valid data.
__forceinline__ __device__ void ConvertInclusiveToExclusiveRow(
    KernelArrayView out,
    KernelArrayView shared_temp,
    int row_index,
    int col_index
) {
  if (col_index == 0) {
    out.At(row_index, col_index) = 0;
  } else {
    out.At(row_index, col_index) = shared_temp.At(row_index, col_index - 1);
  }
}

} // namespace edge_buffer_row_kernels
