#include <gtest/gtest.h>

#include "common/program_args.hpp"

class ProgramArgsTest : public ::testing::Test {};

TEST_F(ProgramArgsTest, DefaultInit) {
  auto program_args = ProgramArgs();
  EXPECT_EQ(program_args.local_n(), 0);
  EXPECT_EQ(program_args.seed(), 1234);
  EXPECT_EQ(program_args.backend(), "mpi");
  EXPECT_EQ(program_args.verbose(), false);
  EXPECT_EQ(program_args.full_matrix_size(), 16);
  EXPECT_EQ(program_args.num_tile_rows(), 2);
  EXPECT_EQ(program_args.num_tile_cols(), 2);
  EXPECT_EQ(program_args.orig_argc(), 0);
  EXPECT_EQ(program_args.orig_argv(), nullptr);
}

TEST_F(ProgramArgsTest, Parse) {
  int argc = 5;
  char *argv[] = {
      const_cast<char *>("program_name"),
      const_cast<char *>("-v"),
      const_cast<char *>("2"),
      const_cast<char *>("789"),
      const_cast<char *>("--backend=mpi"),
  };

  auto program_args = ProgramArgs::Parse(argc, argv);

  EXPECT_EQ(program_args.local_n(), 2);
  EXPECT_EQ(program_args.seed(), 789);
  EXPECT_EQ(program_args.backend(), "mpi");
  EXPECT_EQ(program_args.verbose(), true);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}