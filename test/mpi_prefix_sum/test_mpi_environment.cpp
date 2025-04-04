#include "test_cli_utils.hpp"

#include <mpi.h>

#include <gtest/gtest.h>

#include "mpi_prefix_sum/mpi_environment.hpp"

class MpiEnvironmentTest : public ::testing::Test {
protected:

  ArgvBuilder alt_args_ = ArgvBuilder(
      "--local-n 8 --full-matrix-dim 4 4 --seed 42 --backend mpi -v"
  );
  ProgramArgs alt_program_args_ =
      ProgramArgs::Parse(alt_args_.argc(), alt_args_.argv_data());
};

TEST_F(MpiEnvironmentTest, AltArgsInit) {
  MpiEnvironment mpi_environment(alt_program_args_);
}

int main(int argc, char **argv) {
  // MPI_Init(&argc, &argv);
  ::testing::InitGoogleTest(&argc, argv);
  int result = RUN_ALL_TESTS();
  // MPI_Finalize();
  return result;
}