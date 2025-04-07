// ----------------------------------------------------------------------------
// program_args.hpp
//
// Program args definitions.
// ----------------------------------------------------------------------------

#pragma once

#include <algorithm>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "common/logger.hpp"

class ProgramArgs {
public:
  ProgramArgs() = default;
  ProgramArgs(
      int seed,
      std::string backend,
      LogLevel log_level,
      std::vector<int> full_matrix_dim,
      std::vector<int> tile_dim,
      int orig_argc,
      char **orig_argv
  );

  [[nodiscard]] int seed() const { return seed_; }
  [[nodiscard]] const std::string &backend() const { return backend_; }
  [[nodiscard]] LogLevel log_level() const { return log_level_; }
  [[nodiscard]] const std::vector<int> &full_matrix_dim() const {
    return full_matrix_dim_;
  }
  [[nodiscard]] const std::vector<int> &tile_dim() const { return tile_dim_; }

  [[nodiscard]] int orig_argc() const { return orig_argc_; }
  [[nodiscard]] char **orig_argv() const { return orig_argv_; }

  [[nodiscard]] int FullMatrixSize() const {
    return std::accumulate(
        full_matrix_dim_.begin(),
        full_matrix_dim_.end(),
        1,
        std::multiplies<int>()
    );
  }

  [[nodiscard]] std::vector<int> GridDim() const {
    std::vector<int> result(full_matrix_dim_.size());
    std::transform(
        full_matrix_dim_.begin(),
        full_matrix_dim_.end(),
        tile_dim_.begin(),
        result.begin(),
        [](int x, int y) { return x / y; }
    );
    return result;
  }

  [[nodiscard]] int ElementsPerTile() const {
    return std::accumulate(
        tile_dim().begin(),
        tile_dim().end(),
        1,
        std::multiplies<int>()
    );
  }

  void Print() const {
    std::cout << "ProgramArgs:\n"
              << "Full Matrix Dimensions: " << full_matrix_dim_[0] << " x "
              << full_matrix_dim_[1] << "\n"
              << "Tile Dimensions: " << tile_dim_[0] << " x " << tile_dim_[1]
              << std::endl;
  }

  // std::unique_ptr<PrefixSumSolver> MakeSolver() const;

private:
  int seed_ = 1234;
  std::string backend_ = "mpi";
  LogLevel log_level_ = LogLevel::OFF;

  std::vector<int> full_matrix_dim_ = {4, 4};
  std::vector<int> tile_dim_ = {2, 2};

  int orig_argc_ = 0;
  char **orig_argv_ = nullptr;
};
