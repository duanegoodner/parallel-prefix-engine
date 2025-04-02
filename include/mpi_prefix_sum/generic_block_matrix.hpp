#pragma once

#include <cassert>
#include <stdexcept>
#include <vector>

// Class GenericBlockMatrix: Represents a 1-D array as a logical 2-D matrix.
class GenericBlockMatrix {

public:
  GenericBlockMatrix(int num_rows, int num_cols)
      : num_rows_(num_rows)
      , num_cols_(num_cols)
      , data_(num_rows * num_cols) {}

  // Constructor: Initialize with number of rows, columns, and existing data
  GenericBlockMatrix(int num_rows, int num_cols, const std::vector<int> &data)
      : num_rows_(num_rows)
      , num_cols_(num_cols)
      , data_(data) {
    if (data.size() != static_cast<size_t>(num_rows * num_cols)) {
      throw std::invalid_argument("Data size does not match matrix dimensions"
      );
    }
  }
  int &ValueAt(int row, int col) {
    assert(row >= 0 && col >= 0 && row < num_rows_ && col < num_cols_);
    return data_[row * num_cols_ + col];
  }

  const int &ValueAt(int row, int col) const {
    assert(row >= 0 && col >= 0 && row < num_rows_ && col < num_cols_);
    return data_[row * num_cols_ + col];
  }

private:
  int num_rows_;
  int num_cols_;
  std::vector<int> data_;
};