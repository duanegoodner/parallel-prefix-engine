#pragma once

#include <cuda_runtime.h>

#include "cuda_prefix_sum/internal/kernel_array_view.cuh"

namespace row_scan_multi_block {

  /// Kernel Phase 1: Performs per-block prefix scan (row-wise) on chunks of data.
  /// Each thread block is responsible for a chunk of a row.
  /// Writes exclusive scan result to `out_ptr`, and total sum of the chunk to `block_sums_ptr`.
  __global__ void RowScanMultiBlockPhase1(
      const int *__restrict__ in_ptr,
      int *__restrict__ out_ptr,
      int *__restrict__ block_sums_ptr,
      ArraySize2D size,
      int chunk_size
  );

  /// Kernel Phase 2: Adds scanned block sums (from Phase 1.5) to each chunk's
  /// partial scan result. Each block updates one chunk of a row in-place in `out_ptr`.
  __global__ void RowScanMultiBlockPhase2(
      int *__restrict__ out_ptr,
      const int *__restrict__ scanned_block_sums_ptr,
      ArraySize2D size,
      int chunk_size
  );

  // === Helper accessors for kernel indexing ===

  /// Returns the current row being processed by this thread block.
  /// Each row corresponds to blockIdx.y.
  __forceinline__ __device__ int RowIndex() { return blockIdx.y; }

  /// Returns the index of the chunk this block is processing within a row.
  /// Each chunk corresponds to blockIdx.x.
  __forceinline__ __device__ int ChunkIndex() { return blockIdx.x; }

  /// Returns the total number of chunks per row (gridDim.x).
  __forceinline__ __device__ int NumChunks() { return gridDim.x; }

  /// Returns the starting column index in global memory for the current chunk.
  /// This is based on chunk size and chunk index.
  __forceinline__ __device__ int ChunkStart(int chunk_size) {
    return ChunkIndex() * chunk_size;
  }

  /// Returns this thread's position within its chunk (threadIdx.x).
  /// Used to index shared memory or offset within a chunk.
  __forceinline__ __device__ int ColOffset() { return threadIdx.x; }

  /// Returns the global column index this thread is responsible for.
  /// Combines chunk offset with local thread offset.
  __forceinline__ __device__ int GlobalCol(int chunk_size) {
    return ChunkStart(chunk_size) + ColOffset();
  }

  /// Converts an inclusive scan (held in shared memory) to an exclusive scan.
  ///
  /// `shared_temp` must contain a row-wise inclusive scan.
  /// `out` will be written with the exclusive result at the specified row and col.
  ///
  /// Usage assumes: shared_temp has dimensions {1, chunk_size}
  __forceinline__ __device__ void ConvertInclusiveToExclusiveRow(
      RowMajorKernelArrayView out,
      RowMajorKernelArrayView shared_temp,
      int row_index,
      int col_index
  ) {
    if (col_index == 0) {
      out.At(row_index, col_index) = 0;
    } else {
      out.At(row_index, col_index) = shared_temp.At(0, col_index - 1);
    }
  }

} // namespace row_scan_multi_block
