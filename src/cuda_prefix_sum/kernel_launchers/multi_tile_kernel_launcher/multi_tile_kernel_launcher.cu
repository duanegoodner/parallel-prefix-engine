#include <cuda_runtime.h>

#include <iostream>

#include "common/array_size_2d.hpp"
#include "cuda_prefix_sum/internal/hillis_steele_row_kernel.cuh"
#include "cuda_prefix_sum/internal/kernel_array.hpp"
#include "cuda_prefix_sum/internal/kernel_config_utils.cuh"
#include "cuda_prefix_sum/internal/kernel_launch_params.hpp"
#include "cuda_prefix_sum/internal/multi_tile_kernel.cuh"
#include "cuda_prefix_sum/multi_tile_kernel_launcher.cuh"

MultiTileKernelLauncher::MultiTileKernelLauncher(
    const ProgramArgs &program_args
)
    : program_args_{program_args}
    , right_tile_edge_buffers_{{program_args_.FullMatrixSize2D().num_rows, FirstPassGridDim().x }}
    , right_tile_edge_buffers_ps_{{program_args_.FullMatrixSize2D().num_rows, FirstPassGridDim().x }}
    , bottom_tile_edge_buffers_{{FirstPassGridDim().y, program_args_.FullMatrixSize2D().num_cols}}
    , bottom_tile_edge_buffers_ps_{{FirstPassGridDim().y, program_args_.FullMatrixSize2D().num_cols}} {}

void MultiTileKernelLauncher::Launch(const KernelArray &device_array) {
  constexpr size_t kMaxSharedMemBytes = 98304;
  ConfigureSharedMemoryForKernel(FirstPass, kMaxSharedMemBytes);

  // Prepare launch configuration
  dim3 block_dim = FirstPassBlockDim();
  dim3 grid_dim = FirstPassGridDim();
  size_t shared_mem_size = FirstPassSharedMemPerBlock();

  auto launch_params = CreateKernelLaunchParams(device_array, program_args_);

  FirstPass<<<grid_dim, block_dim, shared_mem_size>>>(
      launch_params,
      right_tile_edge_buffers_.View(),
      bottom_tile_edge_buffers_.View()
  );

  CheckErrors();

  LaunchRowWisePrefixSum(
      right_tile_edge_buffers_.d_address(),
      right_tile_edge_buffers_ps_.d_address(),
      right_tile_edge_buffers_.size(),
      FirstPassGridDim().x
  );

  CheckErrors();
}

dim3 MultiTileKernelLauncher::FirstPassGridDim() {

  auto num_block_rows = program_args_.FullMatrixSize2D().num_rows /
                        program_args_.TileSize2D().num_rows;

  auto num_block_cols = program_args_.FullMatrixSize2D().num_cols /
                        program_args_.TileSize2D().num_cols;

  return dim3(num_block_cols, num_block_rows, 1);
}

dim3 MultiTileKernelLauncher::FirstPassBlockDim() {
  auto num_thread_rows = program_args_.TileSize2D().num_rows /
                         program_args_.SubTileSize2D().num_rows;

  auto num_thread_cols = program_args_.TileSize2D().num_cols /
                         program_args_.SubTileSize2D().num_cols;

  return dim3(num_thread_cols, num_thread_rows, 1);
}

size_t MultiTileKernelLauncher::FirstPassSharedMemPerBlock() {

  return static_cast<size_t>(program_args_.TileSize2D().num_rows) *
         static_cast<size_t>(program_args_.TileSize2D().num_cols) *
         sizeof(int);
}

void MultiTileKernelLauncher::CheckErrors() {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }

  cudaDeviceSynchronize();
  cudaError_t sync_err = cudaGetLastError();
  if (sync_err != cudaSuccess) {
    fprintf(stderr, "CUDA sync error: %s\n", cudaGetErrorString(sync_err));
  }
}

void MultiTileKernelLauncher::LaunchRowWisePrefixSum(
    const int *d_input,
    int *d_output,
    ArraySize2D size,
    int chunk_size
) {
  if (size.num_cols <= 1024) {
    dim3 block(size.num_cols);
    dim3 grid(size.num_rows);
    size_t shared_bytes = size.num_cols * sizeof(int);
    RowWiseScanSingleBlock<<<grid, block, shared_bytes>>>(
        d_input,
        d_output,
        size
    );
  } else {
    // Phase 1: scan chunks
    int num_chunks = (size.num_cols + chunk_size - 1) / chunk_size;
    dim3 grid_phase1(num_chunks, size.num_rows);
    dim3 block_phase1(chunk_size);
    size_t shared_bytes = chunk_size * sizeof(int);

    int *d_block_sums;
    cudaMalloc(&d_block_sums, sizeof(int) * size.num_rows * num_chunks);

    RowWiseScanMultiBlockPhase1<<<grid_phase1, block_phase1, shared_bytes>>>(
        d_input,
        d_output,
        d_block_sums,
        size,
        chunk_size
    );

    // Phase 1.5: scan block sums
    int *d_scanned_block_sums;
    cudaMalloc(&d_scanned_block_sums, sizeof(int) * size.num_rows * num_chunks);

    // Recursively scan block sums row-wise
    LaunchRowWisePrefixSum(
        d_block_sums,
        d_scanned_block_sums,
        size,
        chunk_size
    );

    // Phase 2: apply scanned sums
    RowWiseScanMultiBlockPhase2<<<grid_phase1, block_phase1>>>(
        d_output,
        d_scanned_block_sums,
        size,
        chunk_size
    );

    cudaFree(d_block_sums);
    cudaFree(d_scanned_block_sums);
  }
}
