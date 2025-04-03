#include <gtest/gtest.h>

#include "common/program_args.hpp"

#include "mpi_prefix_sum/mpi_cartesian_grid.hpp"
#include "mpi_prefix_sum/mpi_environment.hpp"

class MpiCartesianGridTest : public ::testing::Test {
protected:
  char *argv_storage_[6] = {
      const_cast<char *>("program_name"),
      const_cast<char *>("-v"),
      const_cast<char *>("2"),
      const_cast<char *>("789"),
      const_cast<char *>("--backend=mpi"),
      nullptr // <--- Important!
  };
  int argc_ = 5;
  char **argv_ = argv_storage_;
  ProgramArgs program_args_ = ProgramArgs::Parse(argc_, argv_);

  MpiEnvironment mpi_environment_ = MpiEnvironment(program_args_);
};

TEST_F(MpiCartesianGridTest, DefaultInit) {
  MpiCartesianGrid grid(
      mpi_environment_.rank(),
      program_args_.num_tile_rows(),
      program_args_.num_tile_cols()
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