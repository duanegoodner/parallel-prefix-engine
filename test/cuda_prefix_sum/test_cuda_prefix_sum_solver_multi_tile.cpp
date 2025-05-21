#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "common/logger.hpp"
#include "common/program_args.hpp"

#include "cuda_prefix_sum/cuda_prefix_sum_solver.hpp"
#include "cuda_prefix_sum/internal/kernel_launch_params.hpp"
#include "cuda_prefix_sum/kernel_launcher.hpp"
#include "cuda_prefix_sum/multi_tile_kernel_launcher.cuh"

class CudaPrefixSumSolverMultiTileTest : public ::testing::Test{
protected:
  int random_seed_ = 1234;
  std::string backend_ = "cuda";
  LogLevel log_level_ = LogLevel::OFF;
  std::vector<size_t> full_matrix_dim_ = std::vector<size_t>({4, 40});
  std::vector<size_t> tile_dim_ = std::vector<size_t>({2, 4});
  std::vector<size_t> subtile_dim_ = std::vector<size_t>({2, 2});
  std::string kernel_ = "multi_tile";
  int orig_argc_ = 1;
  char **orig_argv_ = nullptr;

  ProgramArgs program_args_ = ProgramArgs(
      random_seed_,
      backend_,
      log_level_,
      full_matrix_dim_,
      tile_dim_,
      subtile_dim_,
      kernel_,
      orig_argc_,
      orig_argv_
  );

  std::unique_ptr<KernelLauncher> multi_tile_launcher_ =
      std::make_unique<MultiTileKernelLauncher>(program_args_);
};

TEST_F(CudaPrefixSumSolverMultiTileTest, Init) {
  CudaPrefixSumSolver multi_tile_solver(
      program_args_,
      std::move(multi_tile_launcher_)
  );

  multi_tile_solver.PrintFullMatrix();
}

TEST_F(CudaPrefixSumSolverMultiTileTest, Compute) {
  CudaPrefixSumSolver multi_tile_solver(
      program_args_,
      std::move(multi_tile_launcher_)
  );

  std::cout << "Before computation:";
  multi_tile_solver.PrintFullMatrix();
  multi_tile_solver.Compute();
  std::cout << std::endl;
  std::cout << "After computation:";
  multi_tile_solver.PrintFullMatrix();
}





int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}