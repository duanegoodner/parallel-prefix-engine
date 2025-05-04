#include <gtest/gtest.h>

#include "common/program_args.hpp"

#include "cuda_prefix_sum/cuda_prefix_sum_warp_solver.hpp"

class CudaPrefixSumWarpSolverTest : public ::testing::Test {};

// TEST_F(CudaPrefixSumWarpSolverTest, SimpleOnes) {
//   CudaPrefixSumWarpSolver solver;
//   std::vector<int> input(1024, 1);
//   std::vector<int> output = solver.Run(input);

//   ASSERT_EQ(output.back(), 1024);
// }

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}