#include <gtest/gtest.h>

#include "common/program_args.hpp"

#include "mpi_prefix_sum/internal/mpi_cartesian_grid.hpp"
#include "mpi_prefix_sum/internal/mpi_environment.hpp"

class MpiCartesianGridTest : public ::testing::Test {
protected:

  std::vector<int> full_matrix_dim_ = std::vector<int>({6, 6});
  std::vector<int> grid_dim_ = std::vector<int>({2, 2});
  std::vector<int> tile_dim_ = std::vector<int>({3, 3});
  std::optional<std::vector<int>> maybe_subtile_dim = std::nullopt;
  std::optional<std::string> maybe_kernel = std::nullopt;

  ProgramArgs program_args_ = ProgramArgs(
      1234,
      "mpi",
      LogLevel::OFF,
      full_matrix_dim_,
      tile_dim_,
      maybe_subtile_dim,
      maybe_kernel,
      1,
      nullptr
  );

  MpiEnvironment mpi_environment_ = MpiEnvironment(program_args_);
};

TEST_F(MpiCartesianGridTest, DefaultInit) {
  MpiCartesianGrid grid(
      mpi_environment_.rank(),
      program_args_.TileGridDim()[0],
      program_args_.TileGridDim()[1]
  );

  std::cout << "Total number of ranks is: " << grid.size() << std::endl;
  std::cout << "Coords of rank " << grid.rank() << ": (" << grid.proc_row()
            << ", " << grid.proc_col() << ")" << std::endl;

  EXPECT_EQ(grid.num_rows(), 2);
  EXPECT_EQ(grid.num_cols(), 2);
  EXPECT_EQ(grid.proc_row(), grid.rank() / 2);
  EXPECT_EQ(grid.proc_col(), grid.rank() % 2);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}