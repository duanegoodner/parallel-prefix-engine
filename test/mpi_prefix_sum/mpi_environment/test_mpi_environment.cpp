#include <mpi.h>

#include <gtest/gtest.h>

#include "mpi_prefix_sum/internal/mpi_environment.hpp"

class MpiEnvironmentTest : public ::testing::Test {
protected:

  std::vector<size_t> full_matrix_dim_ = std::vector<size_t>({6, 6});
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
};

TEST_F(MpiEnvironmentTest, AltArgsInit) {
  MpiEnvironment mpi_environment(program_args_);
}

int main(int argc, char **argv) {
  // MPI_Init(&argc, &argv);
  ::testing::InitGoogleTest(&argc, argv);
  int result = RUN_ALL_TESTS();
  // MPI_Finalize();
  return result;
}