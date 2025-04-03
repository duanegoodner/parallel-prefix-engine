#include <mpi.h>

#include <gtest/gtest.h>

#include "mpi_prefix_sum/mpi_environment.hpp"

class MpiEnvironmentTest : public ::testing::Test {
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
};

TEST_F(MpiEnvironmentTest, DefaultInit) {
  MpiEnvironment mpi_environment(program_args_);

  if (mpi_environment.rank() == 0) {
    std::cout << "Total number of ranks is: " << mpi_environment.size()
              << std::endl;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  std::cout << "Hello from Rank: " << mpi_environment.rank() << std::endl;
}

int main(int argc, char **argv) {
  // MPI_Init(&argc, &argv);
  ::testing::InitGoogleTest(&argc, argv);
  int result = RUN_ALL_TESTS();
  // MPI_Finalize();
  return result;
}