#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <vector>

#include "common/array_size_2d.hpp"

#include "cuda_prefix_sum/internal/kernel_array.hpp"
#include "cuda_prefix_sum/internal/kernel_launch_params.hpp"
#include "cuda_prefix_sum/internal/sub_tile_kernels.cuh"

namespace sk = subtile_kernels;

RowMajorKernelArray PrepareRowMajorKernelArray(
    const std::vector<int> &host_vector,
    ArraySize2D array_size
) {
  RowMajorKernelArray kernel_array{array_size};
  cudaMemcpy(
      kernel_array.d_address(),
      host_vector.data(),
      array_size.num_rows * array_size.num_cols * sizeof(int),
      cudaMemcpyHostToDevice
  );
  return kernel_array;
}

void RunSingleTileTest(
    const std::vector<int> &host_vector,
    ArraySize2D array_size,
    ArraySize2D sub_tile_size
) {

  auto kernel_array = PrepareRowMajorKernelArray(host_vector, array_size);

  dim3 grid_dim{1, 1, 1};
  dim3 block_dim{
      static_cast<uint32_t>(array_size.num_cols / sub_tile_size.num_cols),
      static_cast<uint32_t>(array_size.num_rows / sub_tile_size.num_rows),
      1
  };
  size_t shared_mem_size{
      array_size.num_rows * array_size.num_cols * sizeof(int)
  };

  auto tile_size = array_size;

  KernelLaunchParams launch_params{
      kernel_array.View(),
      tile_size,
      sub_tile_size
  };

  kernel_array.DebugPrintOnHost("Before prefix sum");

  sk::SingleTileKernel<<<grid_dim, block_dim, shared_mem_size, 0>>>(
      launch_params
  );

  kernel_array.DebugPrintOnHost("After prefix sum");
}

void RunMultiTileLocalPrefixSumTest(
    const std::vector<int> &host_vector,
    ArraySize2D array_size,
    ArraySize2D tile_size,
    ArraySize2D sub_tile_size
) {
  auto kernel_array = PrepareRowMajorKernelArray(host_vector, array_size);
  dim3 grid_dim{
      static_cast<uint32_t>(array_size.num_cols / tile_size.num_cols),
      static_cast<uint32_t>(array_size.num_rows / tile_size.num_rows),
      1
  };
  dim3 block_dim{
      static_cast<uint32_t>(tile_size.num_cols / sub_tile_size.num_cols),
      static_cast<uint32_t>(tile_size.num_rows / sub_tile_size.num_rows),
      1
  };
  size_t shared_mem_size{
      array_size.num_rows * array_size.num_cols * sizeof(int)
  };
  KernelLaunchParams launch_params{
      kernel_array.View(),
      tile_size,
      sub_tile_size
  };

  RowMajorKernelArray right_edge_buffers{{array_size.num_rows, grid_dim.x}};
  RowMajorKernelArray bottom_edge_buffers{{grid_dim.y, array_size.num_cols}};

  kernel_array.DebugPrintOnHost("Before running kernel");

  sk::MultiTileKernel<<<grid_dim, block_dim, shared_mem_size>>>(
      launch_params,
      right_edge_buffers.View(),
      bottom_edge_buffers.View()
  );

  kernel_array.DebugPrintOnHost("After running kernel");
  right_edge_buffers.DebugPrintOnHost("Right edges buffer");
  bottom_edge_buffers.DebugPrintOnHost("Bottom edges buffer");

  RowMajorKernelArray right_edge_buffers_ps{{array_size.num_rows, grid_dim.x}};
  RowMajorKernelArray bottom_edge_buffers_ps{{array_size.num_cols, grid_dim.y}};
}

class SubTileKernelsTest : public ::testing::Test {

protected:
};

TEST_F(SubTileKernelsTest, SingleTileOnes8x8) {
  ArraySize2D array_size{8, 8};
  ArraySize2D sub_tile_size{2, 2};
  auto host_vector =
      std::vector<int>(array_size.num_rows * array_size.num_cols, 1);
  RunSingleTileTest(host_vector, array_size, sub_tile_size);
}

TEST_F(SubTileKernelsTest, SingleTileOnes12x12) {
  ArraySize2D array_size{12, 12};
  ArraySize2D sub_tile_size{2, 2};
  auto host_vector =
      std::vector<int>(array_size.num_rows * array_size.num_cols, 1);
  RunSingleTileTest(host_vector, array_size, sub_tile_size);
}

TEST_F(SubTileKernelsTest, SingleTileArrayA) {
  std::vector<int>
      host_vector{-7, -1, 2, 6, 5, 7, -5, -7, 9, -7, 7, -8, 3, -4, 4, 6};
  ArraySize2D array_size{4, 4};
  ArraySize2D sub_tile_size{2, 2};

  RunSingleTileTest(host_vector, array_size, sub_tile_size);
}

TEST_F(SubTileKernelsTest, SingleTileArrayB) {
  std::vector<int>
      host_vector{-2, 2, 5, 5, -5, -7, 6, 6, -3, -10, 0, -1, -3, -9, 1, -9};
  ArraySize2D array_size{4, 4};
  ArraySize2D sub_tile_size{2, 2};

  RunSingleTileTest(host_vector, array_size, sub_tile_size);
}

TEST_F(SubTileKernelsTest, SingleTileArrayC) {
  std::vector<int>
      host_vector{0, -2, -10, -10, -3, 4, 2, 7, 8, -3, 3, -1, -4, -1, 1, -2};
  ArraySize2D array_size{4, 4};
  ArraySize2D sub_tile_size{2, 2};

  RunSingleTileTest(host_vector, array_size, sub_tile_size);
}

TEST_F(SubTileKernelsTest, SingleTileArrayD) {
  std::vector<int>
      host_vector{5, -5, 7, -6, -9, 9, -3, -8, -3, 0, 5, 5, 7, 1, -2, 9};
  ArraySize2D array_size{4, 4};
  ArraySize2D sub_tile_size{2, 2};

  RunSingleTileTest(host_vector, array_size, sub_tile_size);
}

TEST_F(SubTileKernelsTest, MultiTileLocalOnes) {
  ArraySize2D array_size{8, 8};
  ArraySize2D tile_size{4, 4};
  ArraySize2D sub_tile_size{2, 2};
  auto host_vector =
      std::vector<int>(array_size.num_rows * array_size.num_cols, 1);

  RunMultiTileLocalPrefixSumTest(
      host_vector,
      array_size,
      tile_size,
      sub_tile_size
  );
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}