#include "test_cli_utils.hpp"

#include <mpi.h>

#include <gtest/gtest.h>

#include "mpi_prefix_sum/mpi_environment.hpp"

class MpiEnvironmentTest : public ::testing::Test {
protected:
  // char *argv_storage_[4] = {
  //     const_cast<char *>("program_name"),
  //     const_cast<char *>("-v"),
  //     // const_cast<char *>("2"),
  //     // const_cast<char *>("789"),
  //     const_cast<char *>("--backend=mpi"),
  //     nullptr // <--- Important!
  // };
  // int argc_ = 3;
  // char **argv_ = argv_storage_;
  // ProgramArgs program_args_ = ProgramArgs::Parse(argc_, argv_);

  ArgvBuilder alt_args_ = ArgvBuilder(
      "--local-n 8 --full-matrix-dim 4 4 --seed 42 --backend mpi -v"
  );
  ProgramArgs alt_program_args_ =
      ProgramArgs::Parse(alt_args_.argc(), alt_args_.argv_data());
};

// TEST_F(MpiEnvironmentTest, DefaultInit) {
//   MpiEnvironment mpi_environment(program_args_);

//   if (mpi_environment.rank() == 0) {
//     std::cout << "Total number of ranks is: " << mpi_environment.size()
//               << std::endl;
//   }
//   MPI_Barrier(MPI_COMM_WORLD);
//   std::cout << "Hello from Rank: " << mpi_environment.rank() << std::endl;
// }

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