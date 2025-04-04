#include "test_cli_utils.hpp"

#include <gtest/gtest.h>

#include "common/matrix_init.hpp"
#include "common/program_args.hpp"

#include "mpi_prefix_sum/mpi_cartesian_grid.hpp"
#include "mpi_prefix_sum/mpi_environment.hpp"
#include "mpi_prefix_sum/mpi_tile_info_distributor.hpp"
#include "mpi_prefix_sum/prefix_sum_block_matrix.hpp"

class MpiTileInfoDistributorTest : public ::testing::Test {
protected:
  // char *argv_storage_[6] = {
  //     const_cast<char *>("program_name"),
  //     const_cast<char *>("-v"),
  //     const_cast<char *>("2"),
  //     const_cast<char *>("789"),
  //     const_cast<char *>("--backend=mpi"),
  //     nullptr // <--- Important!
  // };
  // int argc_ = 5;
  // char **argv_ = argv_storage_;
  // ProgramArgs program_args_ = ProgramArgs::Parse(argc_, argv_);

  ArgvBuilder args_ = ArgvBuilder(
      "--local-n 8 --full-matrix-dim 4 4 --seed 42 --backend mpi -v"
  );

  ProgramArgs program_args_ =
      ProgramArgs::Parse(args_.argc(), args_.argv_data());

  MpiEnvironment mpi_environment_ = MpiEnvironment(program_args_);

  MpiCartesianGrid grid_ = MpiCartesianGrid(
      mpi_environment_.rank(),
      program_args_.num_tile_rows(),
      program_args_.num_tile_cols()
  );
};

TEST_F(MpiTileInfoDistributorTest, DistributeFullMatrix) {

  // MpiEnvironment mpi_environment(alt_program_args_);
  PrefixSumBlockMatrix tile(2, 2);
  MpiTileInfoDistributor distributor(tile, grid_);

  std::vector<int> full_matrix;
  PrefixSumBlockMatrix full_block_matrix;

  if (grid_.rank() == 0) {
    auto full_matrix = GenerateRandomMatrix<int>(
        program_args_.full_matrix_size(),
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