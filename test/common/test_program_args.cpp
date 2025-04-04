#include <gtest/gtest.h>

#include <sstream>
#include <string>
#include <vector>

#include "common/program_args.hpp"

#include "test_cli_utils.hpp"

class ProgramArgsTest : public ::testing::Test {};

TEST_F(ProgramArgsTest, DefaultInit) {
  auto program_args = ProgramArgs();
  EXPECT_EQ(program_args.local_n(), 2);
  EXPECT_EQ(program_args.seed(), 1234);
  EXPECT_EQ(program_args.backend(), "mpi");
  EXPECT_EQ(program_args.verbose(), false);
  EXPECT_EQ(program_args.full_matrix_size(), 16);
  EXPECT_EQ(program_args.num_tile_rows(), 2);
  EXPECT_EQ(program_args.num_tile_cols(), 2);
  EXPECT_EQ(program_args.orig_argc(), 0);
  EXPECT_EQ(program_args.orig_argv(), nullptr);
}

TEST_F(ProgramArgsTest, ParseWithArgs) {
  ArgvBuilder args(
      "--local-n 8 --full-matrix-dim 4 4 --seed 42 --backend mpi -v"
  );

  for (int i = 0; i < args.argc(); ++i) {
    std::cout << "argv[" << i << "] = \"" << args.argv_data()[i] << "\"\n";
  }

  auto program_args = ProgramArgs::Parse(args.argc(), args.argv_data());

  EXPECT_EQ(program_args.local_n(), 8);
  EXPECT_EQ(program_args.seed(), 42);
  EXPECT_EQ(program_args.backend(), "mpi");
  EXPECT_EQ(program_args.verbose(), true);
  EXPECT_EQ(program_args.full_matrix_size(), 16); // 4 * 4
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
