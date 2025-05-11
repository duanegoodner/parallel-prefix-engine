
#include <gtest/gtest.h>

#include <sstream>
#include <string>
#include <vector>

#include "common/program_args.hpp"

class ProgramArgsTest : public ::testing::Test {};

TEST_F(ProgramArgsTest, DefaultInit) {
  auto program_args = ProgramArgs();
  EXPECT_EQ(program_args.seed(), 1234);
  EXPECT_EQ(program_args.backend(), "mpi");
  EXPECT_EQ(program_args.log_level(), LogLevel::OFF);
  EXPECT_EQ(program_args.FullMatrixSize(), 16);
  EXPECT_EQ(program_args.GridDim()[0], 2);
  EXPECT_EQ(program_args.GridDim()[1], 2);
  EXPECT_EQ(program_args.orig_argc(), 0);
  EXPECT_EQ(program_args.orig_argv(), nullptr);
}

TEST_F(ProgramArgsTest, TestInit) {

  std::vector<int> full_matrix_dim_ = std::vector<int>({6, 6});
  std::vector<int> grid_dim_ = std::vector<int>({2, 2});
  std::vector<int> tile_dim_ = std::vector<int>({3, 3});
  std::optional<std::vector<int>> maybe_subtile_dim = std::nullopt;
  std::optional<std::string> maybe_kernel = std::nullopt;


  ProgramArgs program_args = ProgramArgs(
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

  EXPECT_EQ(program_args.seed(), 1234);
  EXPECT_EQ(program_args.backend(), "mpi");
  EXPECT_EQ(program_args.log_level(), LogLevel::OFF);
  EXPECT_EQ(program_args.FullMatrixSize(), 36); // 4 * 4
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
