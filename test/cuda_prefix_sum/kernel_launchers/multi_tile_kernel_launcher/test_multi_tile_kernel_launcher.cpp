#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "common/logger.hpp"
#include "common/program_args.hpp"

#include "cuda_prefix_sum/internal/kernel_launch_params.hpp"
#include "cuda_prefix_sum/multi_tile_kernel_launcher.cuh"

class MultiTileKernelLauncherTest : public ::testing::Test {
protected:
  int random_seed_ = 1234;
  std::string backend_ = "cuda";
  LogLevel log_level_ = LogLevel::OFF;
  std::vector<int> full_matrix_dim_ = std::vector<int>({8, 8});
  std::vector<int> tile_dim_ = std::vector<int>({2, 2});
  std::vector<int> subtile_dim_ = std::vector<int>({2, 2});
  std::string kernel_ = "single_tile";
  int orig_argc_ = 1;
  char **orig_argv_ = nullptr;

};

TEST_F(MultiTileKernelLauncherTest, Init) {}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}