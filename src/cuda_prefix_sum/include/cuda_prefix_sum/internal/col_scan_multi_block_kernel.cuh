#pragma once

#include <cuda_runtime.h>

#include "cuda_prefix_sum/internal/kernel_array_view.cuh"

namespace col_scan_multi_block {

__global__ void ColScanMultiBlockPhase1(
    const int* __restrict__ in_ptr,
    int* __restrict__ out_ptr,
    int* __restrict__ block_sums_ptr,
    ArraySize2D size,
    int chunk_size
);

__global__ void ColScanMultiBlockPhase2(
    int* __restrict__ out_ptr,
    const int* __restrict__ scanned_block_sums_ptr,
    ArraySize2D size,
    int chunk_size
);

// Index of the column being processed (each block in X handles one column)
__forceinline__ __device__ int ColIndex() { return blockIdx.x; }

// Index of chunk (each block in Y processes one chunk of rows)
__forceinline__ __device__ int ChunkIndex() { return blockIdx.y; }

__forceinline__ __device__ int NumChunks() { return gridDim.y; }

// Thread's offset into the chunk (threadIdx.x is row within chunk)
__forceinline__ __device__ int RowOffsetInChunk() { return threadIdx.x; }

// Absolute row index in global memory
__forceinline__ __device__ int GlobalRow(int chunk_size) {
  return ChunkIndex() * chunk_size + RowOffsetInChunk();
}

// Convert inclusive to exclusive
__forceinline__ __device__ void ConvertInclusiveToExclusiveColumn(
    KernelArrayView out,
    KernelArrayView shared_temp,
    int global_row,
    int local_row
) {
  out.At(global_row, ColIndex()) =
      (local_row == 0) ? 0 : shared_temp.At(local_row - 1, 0);
}

}  // namespace col_scan_multi_block
