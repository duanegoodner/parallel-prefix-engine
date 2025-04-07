#include "test_cli_utils.hpp"

#include <mpi.h>

#include <gtest/gtest.h>

#include "common/program_args.hpp"

#include "mpi_prefix_sum/mpi_prefix_sum_solver.hpp"

class MpiPrefixSumSolverTest : public ::testing::Test {
protected:
  // ArgvBuilder args_ = ArgvBuilder("-f 6 6 -g 2 2 -t 3 3");
  // ProgramArgs program_args_ =
  //     ProgramArgs::Parse(args_.argc(), args_.argv_data());

  std::vector<int> full_matrix_dim_ = std::vector<int>({6, 6});
  std::vector<int> grid_dim_ = std::vector<int>({2, 2});
  std::vector<int> tile_dim_ = std::vector<int>({3, 3});

  ProgramArgs program_args_ = ProgramArgs(
      1234,
      "mpi",
      false,
      full_matrix_dim_,
      tile_dim_,
      1,
      nullptr
  );

  MpiPrefixSumSolver mpi_prefix_sum_solver_ =
      MpiPrefixSumSolver(program_args_);
};

TEST_F(MpiPrefixSumSolverTest, Compute) {

  if (mpi_prefix_sum_solver_.Rank() == 0) {
    std::cout << "Before computation:" << std::endl;
  }

  mpi_prefix_sum_solver_.PrintFullMatrix();

  if (mpi_prefix_sum_solver_.Rank() == 0) {
    std::cout << std::endl;
  }

  mpi_prefix_sum_solver_.Compute();

  if (mpi_prefix_sum_solver_.Rank() == 0) {
    std::cout << "After computation:" << std::endl;
  }

  mpi_prefix_sum_solver_.PrintFullMatrix();
}

// TEST_F(MpiPrefixSumSolverTest, Functonality) {
//   if (mpi_prefix_sum_solver_.Rank() == 0) {
//     std::cout << "Rank 0 will subdivide and distribute the following
//     matrix:"
//               << std::endl;
//     mpi_prefix_sum_solver_.PrintFullMatrix();
//     std::cout << std::endl;
//   }

//   mpi_prefix_sum_solver_.DistributeSubMatrices();

//   MPI_Barrier(MPI_COMM_WORLD);

//   std::cout << "Rank " << mpi_prefix_sum_solver_.Rank()
//             << " received sub-matrix." << std::endl;
//   mpi_prefix_sum_solver_.PrintAssignedMatrix();
//   std::cout << std::endl;

//   mpi_prefix_sum_solver_.ComputeAndShareAssigned();

//   MPI_Barrier(MPI_COMM_WORLD);

//   std::cout << "After local compute + sharing at Rank "
//             << mpi_prefix_sum_solver_.Rank() << std::endl;

//   mpi_prefix_sum_solver_.PrintAssignedMatrix();
//   std::cout << std::endl;

//   mpi_prefix_sum_solver_.CollectSubMatrices();
//   if (mpi_prefix_sum_solver_.Rank() == 0) {
//     std::cout << "After collecting sub-matrices at Rank 0:" << std::endl;
//     mpi_prefix_sum_solver_.PrintFullMatrix();
//   }
// }

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}