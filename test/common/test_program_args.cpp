#include <gtest/gtest.h>

#include <sstream>
#include <string>
#include <vector>

#include "common/program_args.hpp"

struct ArgvBuilder {
  std::vector<std::string> args_str;
  std::vector<char *> argv;

  ArgvBuilder(const std::string &cmdline) {
    std::istringstream iss(cmdline);
    std::string token;

    argv.push_back(const_cast<char *>("test_program"));

    while (iss >> token) {
      args_str.push_back(token);
    }

    // Store .data() after args_str is fully built to avoid invalidation
    for (auto &arg : args_str) {
      argv.push_back(const_cast<char *>(arg.data())); // safe + linter-friendly
    }
  }

  int argc() const { return static_cast<int>(argv.size()); }
  char **argv_data() { return argv.data(); }
};

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
      "--local-n 8 --full-matrix-dim 4 4 --seed 42 --backend cuda -v"
  );

  for (int i = 0; i < args.argc(); ++i) {
    std::cout << "argv[" << i << "] = \"" << args.argv_data()[i] << "\"\n";
  }

  auto program_args = ProgramArgs::Parse(args.argc(), args.argv_data());

  EXPECT_EQ(program_args.local_n(), 8);
  EXPECT_EQ(program_args.seed(), 42);
  EXPECT_EQ(program_args.backend(), "cuda");
  EXPECT_EQ(program_args.verbose(), true);
  EXPECT_EQ(program_args.full_matrix_size(), 16); // 4 * 4
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
