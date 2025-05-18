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
#include <optional>
#include <string>
#include <vector>

#include "common/array_size_2d.hpp"
#include "common/logger.hpp"



// Non-member operator==
inline bool operator==(const ArraySize2D& lhs, const ArraySize2D& rhs) {
  return lhs.num_rows == rhs.num_rows && lhs.num_cols == rhs.num_cols;
}

inline bool operator!=(const ArraySize2D& lhs, const ArraySize2D& rhs) {
  return !(lhs == rhs);
}

class ProgramArgs {
public:
  ProgramArgs() = default;
  ProgramArgs(
      int seed,
      std::string backend,
      LogLevel log_level,
      std::vector<int> full_matrix_dim,
      std::vector<int> tile_dim,
      std::optional<std::vector<int>> sub_tile_dim,
      std::optional<std::string> cuda_kernel,
      int orig_argc,
      char **orig_argv
  );

  int seed() const { return seed_; }

  const std::string &backend() const { return backend_; }

  LogLevel log_level() const { return log_level_; }

  const std::vector<int> &full_matrix_dim() const { return full_matrix_dim_; }

  const std::vector<int> &tile_dim() const { return tile_dim_; }

  const std::optional<std::vector<int>> &sub_tile_dim() const {
    return sub_tile_dim_;
  }

  std::optional<std::string> cuda_kernel() const { return cuda_kernel_; }

  int orig_argc() const { return orig_argc_; }
  char **orig_argv() const { return orig_argv_; }

  int FullMatrixSize1D() const {
    return std::accumulate(
        full_matrix_dim_.begin(),
        full_matrix_dim_.end(),
        1,
        std::multiplies<int>()
    );
  }

  // Convenience methods to help avoid errors
  ArraySize2D FullMatrixSize2D() const {
    return ArraySize2D{
        .num_rows = full_matrix_dim_[0],
        .num_cols = full_matrix_dim_[1]};
  }
  ArraySize2D TileSize2D() const {
    return ArraySize2D{.num_rows = tile_dim_[0], .num_cols = tile_dim_[1]};
  }
  ArraySize2D SubTileSize2D() const {
    if (!sub_tile_dim_.has_value()) {
      throw std::runtime_error("sub_tile_dim_ is not set.");
    }
    return ArraySize2D{
        .num_rows = sub_tile_dim_.value()[0],
        .num_cols = sub_tile_dim_.value()[1]};
  }

  std::vector<int> TileGridDim() const {
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

  int ElementsPerTile() const {
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

private:
  int seed_ = 1234;
  std::string backend_ = "mpi";
  LogLevel log_level_ = LogLevel::WARNING;

  std::vector<int> full_matrix_dim_ = {4, 4};
  std::vector<int> tile_dim_ = {4, 4};
  std::optional<std::vector<int>> sub_tile_dim_;
  std::optional<std::string> cuda_kernel_;

  int orig_argc_ = 0;
  char **orig_argv_ = nullptr;

  bool IsFullMatrixDimDivisibleByTileDim() {
    for (size_t i = 0; i < full_matrix_dim_.size(); ++i) {
      if (full_matrix_dim_[i] % tile_dim_[i] != 0) {
        return false;
      }
    }
    return true;
  }

  bool TileAndFullMatrixHaveSameNumDims() {
    return full_matrix_dim_.size() == tile_dim_.size();
  }
};
