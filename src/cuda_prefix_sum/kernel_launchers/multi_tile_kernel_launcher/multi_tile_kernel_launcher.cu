#include <cuda_runtime.h>

#include <iostream>
#include <vector>

#include "common/array_size_2d.hpp"

#include "cuda_prefix_sum/internal/col_scan_multi_block_kernel.cuh"
#include "cuda_prefix_sum/internal/col_scan_single_block_kernel.cuh"
#include "cuda_prefix_sum/internal/kernel_array.hpp"
#include "cuda_prefix_sum/internal/kernel_config_utils.cuh"
#include "cuda_prefix_sum/internal/kernel_launch_params.hpp"
#include "cuda_prefix_sum/internal/multi_block_prefix_sum_helpers.cuh"
#include "cuda_prefix_sum/internal/row_scan_multi_block_kernel.cuh"
#include "cuda_prefix_sum/internal/row_scan_single_block_kernel.cuh"
#include "cuda_prefix_sum/internal/row_to_col_injection_kernel.cuh"
#include "cuda_prefix_sum/internal/sub_tile_kernels.cuh"
#include "cuda_prefix_sum/multi_tile_kernel_launcher.cuh"

namespace sk = subtile_kernels;
namespace rsingle = row_scan_single_block;
namespace rmulti = row_scan_multi_block;
namespace rtci = row_to_col_injection;
namespace csingle = col_scan_single_block;
namespace cmulti = col_scan_multi_block;

MultiTileKernelLauncher::MultiTileKernelLauncher(
    const ProgramArgs &program_args
)
    : program_args_{program_args}
    , right_tile_edge_buffers_{{program_args_.FullMatrixSize2D().num_rows, FirstPassGridDim().x }}
    , right_tile_edge_buffers_ps_{{program_args_.FullMatrixSize2D().num_rows, FirstPassGridDim().x }}
    , bottom_tile_edge_buffers_{{FirstPassGridDim().y, program_args_.FullMatrixSize2D().num_cols}}
    , bottom_tile_edge_buffers_ps_{{FirstPassGridDim().y, program_args_.FullMatrixSize2D().num_cols}} {}

void MultiTileKernelLauncher::Launch(const RowMajorKernelArray &device_array) {
  constexpr size_t kMaxSharedMemBytes = 98304;
  ConfigureSharedMemoryForKernel(sk::MultiTileKernel, kMaxSharedMemBytes);

  // Prepare launch configuration
  dim3 block_dim = FirstPassBlockDim();
  dim3 grid_dim = FirstPassGridDim();
  size_t shared_mem_size = FirstPassSharedMemPerBlock();

  auto launch_params = CreateKernelLaunchParams(device_array, program_args_);

  sk::MultiTileKernel<<<grid_dim, block_dim, shared_mem_size>>>(
      launch_params,
      right_tile_edge_buffers_.View(),
      bottom_tile_edge_buffers_.View()
  );

  CheckErrors();

  LaunchRowWisePrefixSum(
      right_tile_edge_buffers_.d_address(),
      right_tile_edge_buffers_ps_.d_address(),
      right_tile_edge_buffers_.size()
  );

  CheckErrors();

  LaunchRowToColInjection();
  CheckErrors();

  LaunchColWisePrefixSum(
      bottom_tile_edge_buffers_.d_address(),
      bottom_tile_edge_buffers_ps_.d_address(),
      bottom_tile_edge_buffers_.size()
  );

  CheckErrors();

  sk::ApplyTileGlobalOffsets<<<FirstPassGridDim(), FirstPassBlockDim()>>>(
      launch_params,
      right_tile_edge_buffers_ps_.ConstView(),
      bottom_tile_edge_buffers_ps_.ConstView()
  );

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
    ArraySize2D size
) {
  if (size.num_cols <= buffer_sum_method_cutoff_) {
    dim3 block(size.num_cols);
    dim3 grid(size.num_rows);
    size_t shared_bytes = size.num_cols * sizeof(int);
    rsingle::RowScanSingleBlockKernel<<<grid, block, shared_bytes>>>(
        d_input,
        d_output,
        size
    );
  } else {
    multi_block_prefix_sum::Launch(
        d_input,
        d_output,
        size,
        mult_block_buffer_sum_chunk_size_,
        /*row_major=*/true,
        rmulti::RowScanMultiBlockPhase1,
        rmulti::RowScanMultiBlockPhase2,
        [this](const int *in, int *out, ArraySize2D sz) {
          LaunchRowWisePrefixSum(in, out, sz);
        }
    );
  }
}

void MultiTileKernelLauncher::LaunchRowToColInjection() {

  constexpr int block_dim_y = 128;

  // Determine how many thread blocks needed along y to cover all rows
  int grid_dim_y =
      (bottom_tile_edge_buffers_.size().num_rows + block_dim_y - 1) /
      block_dim_y;
  
  dim3 block_dim(1, block_dim_y);
  dim3 grid_dim(bottom_tile_edge_buffers_.size().num_cols, grid_dim_y);

  rtci::RowToColInjection<<<grid_dim, block_dim>>>(
      bottom_tile_edge_buffers_.d_address(),
      bottom_tile_edge_buffers_.size(),
      right_tile_edge_buffers_ps_.d_address(),
      right_tile_edge_buffers_ps_.size(),
      program_args_.TileSize2D()
  );
}

void MultiTileKernelLauncher::LaunchColWisePrefixSum(
    const int *d_input,
    int *d_output,
    ArraySize2D bottom_edge_buffer_size
) {
  if (bottom_edge_buffer_size.num_rows <= buffer_sum_method_cutoff_) {
    dim3 block(bottom_edge_buffer_size.num_rows);
    dim3 grid(bottom_edge_buffer_size.num_cols);
    size_t shared_bytes = bottom_edge_buffer_size.num_rows * sizeof(int);
    csingle::ColScanSingleBlockKernel<<<grid, block, shared_bytes>>>(
        d_input,
        d_output,
        bottom_edge_buffer_size,
        right_tile_edge_buffers_ps_.d_address(),
        right_tile_edge_buffers_ps_.size(),
        program_args_.TileSize2D()
    );
  } else {

    multi_block_prefix_sum::Launch(
        d_input,
        d_output,
        bottom_edge_buffer_size,
        mult_block_buffer_sum_chunk_size_,
        /*row_major=*/false,
        cmulti::ColScanMultiBlockPhase1,
        cmulti::ColScanMultiBlockPhase2,
        [this](const int *in, int *out, ArraySize2D sz) {
          LaunchColWisePrefixSum(in, out, sz);
        }

    );
  }
}
