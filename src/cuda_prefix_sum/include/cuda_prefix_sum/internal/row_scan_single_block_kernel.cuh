#pragma once

#include <cuda_runtime.h>

#include "common/array_size_2d.hpp"
#include "cuda_prefix_sum/internal/kernel_array_view.cuh"

namespace row_scan_single_block {

/// Launches a single-block row-wise exclusive prefix sum (scan) for each row
/// of a 2D array using Hillis-Steele.
///
/// This kernel is designed to be launched with one block per row and one thread
/// per column. The input and output pointers (`in_ptr`, `out_ptr`) must point to
/// a contiguous 2D array in **row-major** layout in global memory, with dimensions
/// specified by `size.num_rows` x `size.num_cols`.
///
/// @param in_ptr    Pointer to the input 2D array in global memory
/// @param out_ptr   Pointer to the output 2D array in global memory
/// @param size      The dimensions (rows × cols) of the arrays
__global__ void RowScanSingleBlockKernel(
    const int *__restrict__ in_ptr,
    int *__restrict__ out_ptr,
    ArraySize2D size
);

/// Returns the index of the current row being processed in the input/output
/// 2D arrays located in global memory (row-major layout).
__forceinline__ __device__ int RowIndex() { return blockIdx.x; }

/// Returns the column index within the current row.
/// This also corresponds to the thread index within the block and maps directly
/// to:
///   - the column in the global memory array (i.e., `in[row][col]`)
///   - the index of a thread's assigned element in the shared memory buffer
__forceinline__ __device__ int ColIndex() { return threadIdx.x; }


/// Load one row from global memory into shared memory.
///
/// @param global_array The input array in global memory
/// @param shared_row   A single-row temporary buffer in shared memory
/// @param row_index    The row of global_array to load (usually `blockIdx.x`)
__forceinline__ __device__ void LoadRowToShared(
    RowMajorKernelArrayViewConst global_array,
    RowMajorKernelArrayView shared_row,
    int row_index
) {
  shared_row.At(0, ColIndex()) = global_array.At(row_index, ColIndex());
}

/// Perform an **inclusive** Hillis-Steele scan over the shared row.
///
/// @param shared_row The temporary shared memory buffer
/// @param num_cols   Number of columns to scan
__forceinline__ __device__ void InclusiveScanHillisSteele(
    RowMajorKernelArrayView shared_row,
    int num_cols
) {
  for (int offset = 1; offset < num_cols; offset *= 2) {
    int val = (ColIndex() >= offset) ? shared_row.At(0, ColIndex() - offset) : 0;
    __syncthreads();
    shared_row.At(0, ColIndex()) += val;
    __syncthreads();
  }
}

/// Convert an **inclusive** scan result to **exclusive** in-place.
///
/// @param shared_row The result of an inclusive scan
/// @param output     The output array in global memory
/// @param row_index  The destination row in the output array
__forceinline__ __device__ void ConvertToExclusiveScan(
    RowMajorKernelArrayView shared_row,
    RowMajorKernelArrayView output,
    int row_index
) {
  int col = ColIndex();
  output.At(row_index, col) = (col == 0) ? 0 : shared_row.At(0, col - 1);
}

}  // namespace row_scan_single_block
