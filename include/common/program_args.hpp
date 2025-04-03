// ----------------------------------------------------------------------------
// program_args.hpp
//
// Program args definitions.
// ----------------------------------------------------------------------------

#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "common/prefix_sum_solver.hpp"

class ProgramArgs {
public:
  ProgramArgs() = default;
  ProgramArgs(
      int local_n,
      int seed,
      std::string backend,
      bool verbose,
      std::vector<int> full_matrix_dim,
      std::vector<int> grid_dim,
      int orig_argc,
      char **orig_argv
  );

  static ProgramArgs Parse(int argc, char *const argv[]);

  [[nodiscard]] int local_n() const { return local_n_; }
  [[nodiscard]] int seed() const { return seed_; }
  [[nodiscard]] const std::string &backend() const { return backend_; }
  [[nodiscard]] bool verbose() const { return verbose_; }
  [[nodiscard]] int full_matrix_size() const { return full_matrix_size_; }
  [[nodiscard]] int num_tile_rows() const { return num_tile_rows_; }
  [[nodiscard]] int num_tile_cols() const { return num_tile_cols_; }
  [[nodiscard]] const std::vector<int> &full_matrix_dim() const { return full_matrix_dim_; }
  [[nodiscard]] const std::vector<int> &grid_dim() const { return grid_dim_; }
  [[nodiscard]] int orig_argc() const { return orig_argc_; }
  [[nodiscard]] char **orig_argv() const { return orig_argv_; }

  std::unique_ptr<PrefixSumSolver> MakeSolver() const;

private:
  int local_n_ = 2;
  int seed_ = 1234;
  std::string backend_ = "mpi";
  bool verbose_ = false;

  std::vector<int> full_matrix_dim_ = {4, 4};
  std::vector<int> grid_dim_ = {2, 2};

  int full_matrix_size_ = 16;
  int num_tile_rows_ = 2;
  int num_tile_cols_ = 2;

  int orig_argc_ = 0;
  char **orig_argv_ = nullptr;
};
