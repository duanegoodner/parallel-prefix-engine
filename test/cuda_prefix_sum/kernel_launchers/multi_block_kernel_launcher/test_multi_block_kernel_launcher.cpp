#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

#include "cuda_prefix_sum/multi_block_kernel_launcher.cuh"
#include "cuda_prefix_sum/internal/kernel_launch_params.hpp"
#include "common/program_args.hpp"

class MultiBlockKernelLauncherTest : public ::testing::Test {
    protected:
    std::vector<int> full_matrix_dim_ = std::vector<int>({8, 8});
    std::vector<int> tile_dim_ = std::vector<int>({2, 2});
    std::vector<int> subtile_dim_ = std::vector<int>({2, 2});

    
};

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}