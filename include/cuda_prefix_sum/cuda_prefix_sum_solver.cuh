// cuda_prefix_sum_solver.cuh
#pragma once
#include <cuda_runtime.h>

#include "common/program_args.hpp"

void LaunchPrefixSumKernel(
    int *d_data,
    int rows_per_block,
    int cols_per_block,
    int blocks_per_row,
    int blocks_per_col,
    cudaStream_t stream
);

struct KernelParams {
  int tile_dim_x;
  int tile_dim_y;
  int full_matrix_dim_x;
  int full_matrix_dim_y;
  int num_tile_rows;
  int num_tile_cols;

  KernelParams(const ProgramArgs &program_args)
      : tile_dim_x(program_args.tile_dim()[0])
      , tile_dim_y(program_args.tile_dim()[1])
      , full_matrix_dim_x(program_args.full_matrix_dim()[0])
      , full_matrix_dim_y(program_args.full_matrix_dim()[1])
      , num_tile_rows(program_args.GridDim()[0])
      , num_tile_cols(program_args.GridDim()[1]) {}

  void Print() const {
    std::cout << "tile_dim_x = " << tile_dim_x << "\n"
              << "tile_dim_y = " << tile_dim_y << "\n"
              << "full_matrix_dim_x" << full_matrix_dim_x << "\n"
              << "full_matrix_dimy" << full_matrix_dim_y << "\n"
              << "num_tile_rows" << num_tile_rows << "\n"
              << "num_tile_cols" << num_tile_cols << std::endl;
   }
};

void LaunchPrefixSumKernelNew(
    int *d_data,
    const ProgramArgs& program_args,
    cudaStream_t cudaStream
);
