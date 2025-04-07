#include "test_cli_utils.hpp"

#include <gtest/gtest.h>

#include "common/program_args.hpp"

#include "cuda_prefix_sum/cuda_prefix_sum_solver.hpp"

class CudaPrefixSumSolverTest : public ::testing::Test {
protected:
  // ArgvBuilder args_ = ArgvBuilder("-f 6 6 -g 2 2 --backend mpi -v");
  // ProgramArgs program_args_ =
  //     ProgramArgs::Parse(args_.argc(), args_.argv_data());

  std::vector<int> full_matrix_dim_ = std::vector<int>({6, 6});
  std::vector<int> grid_dim_ = std::vector<int>({2, 2});
  std::vector<int> tile_dim_ = std::vector<int>({3, 3});

  ProgramArgs program_args_ =
      ProgramArgs(1234, "mpi", LogLevel::OFF, full_matrix_dim_, tile_dim_, 1, nullptr);
};

TEST_F(CudaPrefixSumSolverTest, DefaultInit) {
  CudaPrefixSumSolver cuda_solver{program_args_};

  // std::cout << "Full array size (flattened): " << cuda_solver.Full <<
  // std::endl;

  // std::cout << "Initial array on host:" << std::endl;
  // cuda_solver.PrintFullMatrix();

  std::cout << "Before computation:" << std::endl;
  cuda_solver.PrintFullMatrix();
  cuda_solver.Compute();
  std::cout << "After computation:" << std::endl;
  cuda_solver.PrintFullMatrix();

  std::cout << "end of test" << std::endl;
  // cuda_solver.PrintFullMatrix();
  // std::cout << "pause" << std::endl;
}

// TEST_F(CudaPrefixSumSolverTest, ComputeNew) {
//   CudaPrefixSumSolver cuda_solver{program_args_};
//   cuda_solver.PrintFullMatrix();
//   std::cout << std::endl;
//   cuda_solver.ComputeNew();
//   cuda_solver.PrintFullMatrix();
// }

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}