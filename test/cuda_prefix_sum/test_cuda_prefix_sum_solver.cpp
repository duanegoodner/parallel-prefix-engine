#include "test_cli_utils.hpp"

#include <gtest/gtest.h>

#include "common/program_args.hpp"

#include "cuda_prefix_sum/cuda_prefix_sum_solver.hpp"

class CudaPrefixSumSolverTest : public ::testing::Test {
protected:
  ArgvBuilder args_ = ArgvBuilder("-f 4 4 -g 2 2 -s 42 --backend mpi -v");
  ProgramArgs program_args_ =
      ProgramArgs::Parse(args_.argc(), args_.argv_data());
};

// TEST_F(CudaPrefixSumSolverTest, DefaultInit) {
//   CudaPrefixSumSolver cuda_solver{program_args_};
//   cuda_solver.PrintFullMatrix();
//   std::cout << "pause" << std::endl;
// }

TEST_F(CudaPrefixSumSolverTest, ComputeNew) {
  CudaPrefixSumSolver cuda_solver{program_args_};
  cuda_solver.PrintFullMatrix();
  std::cout << std::endl;
  cuda_solver.ComputeNew();
  cuda_solver.PrintFullMatrix();
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}