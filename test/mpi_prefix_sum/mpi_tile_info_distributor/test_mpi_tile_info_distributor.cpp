#include <gtest/gtest.h>

#include "common/matrix_init.hpp"
#include "common/program_args.hpp"

#include "mpi_prefix_sum/internal/mpi_cartesian_grid.hpp"
#include "mpi_prefix_sum/internal/mpi_environment.hpp"
#include "mpi_prefix_sum/internal/mpi_tile_info_distributor.hpp"
#include "mpi_prefix_sum/internal/prefix_sum_block_matrix.hpp"

class MpiTileInfoDistributorTest : public ::testing::Test {
protected:

  std::vector<size_t> full_matrix_dim_ = std::vector<size_t>({6, 6});
  std::vector<size_t> grid_dim_ = std::vector<size_t>({2, 2});
  std::vector<size_t> tile_dim_ = std::vector<size_t>({3, 3});
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

  MpiEnvironment mpi_environment_ = MpiEnvironment(program_args_);

  MpiCartesianGrid grid_ = MpiCartesianGrid(
      mpi_environment_.rank(),
      program_args_.TileGridDim()[0],
      program_args_.TileGridDim()[1]
  );
};

TEST_F(MpiTileInfoDistributorTest, DistributeFullMatrix) {

  PrefixSumBlockMatrix tile(2, 2);
  MpiTileInfoDistributor distributor(tile, grid_);

  std::vector<int> full_matrix;
  PrefixSumBlockMatrix full_block_matrix;

  if (grid_.rank() == 0) {
    auto full_matrix = GenerateRandomMatrix<int>(
        program_args_.full_matrix_dim()[0],
        program_args_.full_matrix_dim()[1],
        program_args_.seed()
    );
    full_block_matrix = PrefixSumBlockMatrix(4, 4, full_matrix);
  }

  distributor.DistributeFullMatrix(full_block_matrix);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}