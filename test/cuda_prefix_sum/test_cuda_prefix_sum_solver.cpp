#include "test_cli_utils.hpp"

#include <gtest/gtest.h>

#include "common/program_args.hpp"

#include "cuda_prefix_sum/cuda_prefix_sum_solver.cuh"
#include "cuda_prefix_sum/cuda_prefix_sum_solver.hpp"
#include "cuda_prefix_sum/kernel_launch_params.hpp"

void DummyKernelLauncher(KernelLaunchParams kernel_params) { return; }

ProgramArgs GenerateProgramArgsForTest(
    std::vector<int> full_matrix_dim,
    std::vector<int> tile_dim
) {
  return ProgramArgs(
      1234,
      "cuda",
      LogLevel::OFF,
      full_matrix_dim,
      tile_dim,
      "tiled",
      1,
      nullptr
  );
}

class CudaPrefixSumSolverTest : public ::testing::Test {
protected:
  std::vector<int> full_matrix_dim_ = std::vector<int>({4, 4});
  std::vector<int> tile_dim_ = std::vector<int>({2, 2});
  ProgramArgs program_args_ =
      GenerateProgramArgsForTest(full_matrix_dim_, tile_dim_);
};

TEST_F(CudaPrefixSumSolverTest, Init) {

  CudaPrefixSumSolver cuda_solver(program_args_, DummyKernelLauncher);
}

TEST_F(CudaPrefixSumSolverTest, PublicTimer) {
  CudaPrefixSumSolver cuda_solver(program_args_, DummyKernelLauncher);
  cuda_solver.StartTimer();
  cuda_solver.StopTimer();
}

TEST_F(CudaPrefixSumSolverTest, Compute) {

  CudaPrefixSumSolver cuda_solver{program_args_, LaunchPrefixSumKernelTiled};

  std::cout << "Before computation:";
  cuda_solver.PrintFullMatrix();
  cuda_solver.Compute();
  std::cout << std::endl;
  std::cout << "After computation:";
  cuda_solver.PrintFullMatrix();
}

TEST_F(CudaPrefixSumSolverTest, ReportTime) {
  CudaPrefixSumSolver cuda_solver{program_args_, LaunchPrefixSumKernelTiled};
  cuda_solver.StartTimer();
  cuda_solver.Compute();
  cuda_solver.StopTimer();
  cuda_solver.ReportTime();
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}