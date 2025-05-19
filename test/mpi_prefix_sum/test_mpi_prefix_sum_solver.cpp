#include <mpi.h>

#include <gtest/gtest.h>

#include "common/program_args.hpp"

#include "mpi_prefix_sum/mpi_prefix_sum_solver.hpp"

class MpiPrefixSumSolverTest : public ::testing::Test {
protected:
  std::vector<size_t> full_matrix_dim_ = std::vector<size_t>({4, 4});
  // std::vector<size_t> grid_dim_ = std::vector<size_t>({2, 2});
  std::vector<size_t> tile_dim_ = std::vector<size_t>({2, 2});
  std::optional<std::vector<size_t>> maybe_subtile_dim_ = std::nullopt;
  std::optional<std::string> maybe_kernel_ = std::nullopt;

  ProgramArgs program_args_ = ProgramArgs(
      1234,
      "mpi",
      LogLevel::OFF,
      full_matrix_dim_,
      tile_dim_,
      maybe_subtile_dim_,
      maybe_kernel_,
      1,
      nullptr
  );

  MpiPrefixSumSolver mpi_prefix_sum_solver_ =
      MpiPrefixSumSolver(program_args_);
};

TEST_F(MpiPrefixSumSolverTest, Compute) {

  if (mpi_prefix_sum_solver_.Rank() == 0) {
    std::cout << "Before computation:";
  }

  mpi_prefix_sum_solver_.PrintFullMatrix();

  if (mpi_prefix_sum_solver_.Rank() == 0) {
    std::cout << std::endl;
  }

  mpi_prefix_sum_solver_.Compute();

  if (mpi_prefix_sum_solver_.Rank() == 0) {
    std::cout << "After computation:";
  }

  mpi_prefix_sum_solver_.PrintFullMatrix();
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}