// ----------------------------------------------------------------------------
// prefix_sum_block_matrix.hpp
//
// Prefix sum block matrix definitions.
// This header is part of the prefix sum project.
// ----------------------------------------------------------------------------

#pragma once

#include <mpi.h>

#include <cassert>
#include <unordered_map>
#include <vector>

// Class PrefixSumBlockMatrix: Represents a local matrix block in the global 2D
// grid for prefix sum computation.
class PrefixSumBlockMatrix {
public:
  PrefixSumBlockMatrix() = default;
  explicit PrefixSumBlockMatrix(int square_dim);
  explicit PrefixSumBlockMatrix(int num_rows, int num_cols);
  explicit PrefixSumBlockMatrix(
      int num_rows,
      int num_cols,
      std::vector<int> data
  );

  int &ValueAt(int row, int col);
  const int &ValueAt(int row, int col) const;

  void ComputeLocalPrefixSum();

  std::vector<int> &data() { return data_; }
  const std::vector<int> &data() const { return data_; }

  // int local_n() const { return local_n_; }
  int num_rows() const { return num_rows_; }
  int num_cols() const { return num_cols_; }

  std::unordered_map<int, std::vector<int>> SubDivide(
      int rows_per_tile,
      int cols_per_tile
  ) const;

  std::vector<int> ExtractRightEdge() const;
  std::vector<int> ExtractBottomEdge() const;

  void AddRowwiseOffset(const std::vector<int> &offsets);
  void AddColwiseOffset(const std::vector<int> &offsets);

private:
  int num_rows_;
  int num_cols_;
  // int local_n_;
  std::vector<int> data_;
};