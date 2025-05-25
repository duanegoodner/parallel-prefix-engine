#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "common/logger.hpp"
#include "common/program_args.hpp"

#include "cuda_prefix_sum/internal/kernel_launch_params.hpp"
#include "cuda_prefix_sum/multi_tile_kernel_launcher.cuh"

KernelArray PrepareKernelArray(
    const std::vector<int> &host_vector,
    ArraySize2D array_size
) {
  KernelArray kernel_array{array_size};
  cudaMemcpy(
      kernel_array.d_address(),
      host_vector.data(),
      array_size.num_rows * array_size.num_cols * sizeof(int),
      cudaMemcpyHostToDevice
  );
  return kernel_array;
}

class MultiTileKernelLauncherTest : public ::testing::Test {
protected:
  int random_seed_{1234};
  std::string backend_{"cuda"};
  LogLevel log_level_{LogLevel::OFF};
  std::vector<size_t> full_matrix_dim_{12, 12};
  std::vector<size_t> tile_dim_{4, 4};
  std::vector<size_t> sub_tile_dim_{2, 2};
  std::string kernel_{"multi_tile"};
  int orig_argc_{1};
  char **orig_argv_{nullptr};

  ProgramArgs program_args_ = ProgramArgs(
      random_seed_,
      backend_,
      log_level_,
      full_matrix_dim_,
      tile_dim_,
      sub_tile_dim_,
      kernel_,
      orig_argc_,
      orig_argv_
  );
};

TEST_F(MultiTileKernelLauncherTest, AllOnesInput) {
  auto kernel_launcher = MultiTileKernelLauncher(program_args_);
  std::vector<int> host_vector =
      std::vector<int>(program_args_.FullMatrixSize1D(), 1);
  auto kernel_array =
      PrepareKernelArray(host_vector, program_args_.FullMatrixSize2D());
  kernel_launcher.Launch(kernel_array);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}