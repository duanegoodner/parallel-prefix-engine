#include "test_cli_utils.hpp"

#include <gtest/gtest.h>

#include "common/program_args.hpp"

#include "cuda_prefix_sum/cuda_prefix_sum_solver.hpp"

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
      1,
      nullptr
  );
}

class CudaPrefixSumSolverTest : public ::testing::Test {};


TEST_F(CudaPrefixSumSolverTest, FullSize4x4_TileSize2x2) {
  std::vector<int> full_matrix_dim = std::vector<int>({4, 4});
  std::vector<int> tile_dim = std::vector<int>({2, 2});

  auto program_args = GenerateProgramArgsForTest(full_matrix_dim, tile_dim);

  CudaPrefixSumSolver cuda_solver{program_args};

  std::cout << "Before computation:";
  cuda_solver.PrintFullMatrix();
  cuda_solver.Compute();
  std::cout << std::endl;
  std::cout << "After computation:";
  cuda_solver.PrintFullMatrix();
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}