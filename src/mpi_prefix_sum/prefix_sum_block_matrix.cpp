// ----------------------------------------------------------------------------
// prefix_sum_block_matrix.cpp
//
// Prefix sum block matrix implementation.
// ----------------------------------------------------------------------------

#include "mpi_prefix_sum/prefix_sum_block_matrix.hpp"

PrefixSumBlockMatrix::PrefixSumBlockMatrix(int square_dim)
    : num_rows_(square_dim)
    , num_cols_(square_dim)
    , data_(square_dim * square_dim) {}

PrefixSumBlockMatrix::PrefixSumBlockMatrix(int num_rows, int num_cols)
    : num_rows_(num_rows)
    , num_cols_(num_cols)
    , data_(num_rows * num_cols) {}

PrefixSumBlockMatrix::PrefixSumBlockMatrix(
    int num_rows,
    int num_cols,
    std::vector<int> data
)
    : num_rows_(num_rows)
    , num_cols_(num_cols)
    , data_(std::move(data)) {}

int &PrefixSumBlockMatrix::ValueAt(int row, int col) {
  assert(row >= 0 && col >= 0 && row < num_rows_ && col < num_cols_);
  return data_[row * num_cols_ + col];
}

const int &PrefixSumBlockMatrix::ValueAt(int row, int col) const {
  assert(row >= 0 && col >= 0 && row < num_rows_ && col < num_cols_);
  return data_[row * num_cols_ + col];
}

void PrefixSumBlockMatrix::ComputeLocalPrefixSum() {
  for (int row = 0; row < num_rows_; ++row) {
    for (int col = 1; col < num_cols_; ++col) {
      ValueAt(row, col) += ValueAt(row, col - 1);
    }
  }

  for (int col = 0; col < num_cols_; ++col) {
    for (int row = 1; row < num_rows_; ++row) {
      ValueAt(row, col) += ValueAt(row - 1, col);
    }
  }
}

std::unordered_map<int, std::vector<int>> PrefixSumBlockMatrix::SubDivide(
    int tiles_per_row,
    int tiles_per_col
) const {
  // Validate input dimensions
  if (num_rows_ % tiles_per_row != 0 || num_cols_ % tiles_per_col != 0) {
    throw std::invalid_argument("Matrix dimensions must be divisible by "
                                "tiles_per_row and tiles_per_col");
  }

  // Calculate the size of each tile
  int rows_per_tile = num_rows_ / tiles_per_row;
  int cols_per_tile = num_cols_ / tiles_per_col;

  // Map to store the tiles
  std::unordered_map<int, std::vector<int>> tiles;

  // Loop through the matrix and assign elements to the appropriate tile
  for (int row = 0; row < num_rows_; ++row) {
    for (int col = 0; col < num_cols_; ++col) {
      // Determine the tile index
      int tile_row = row / rows_per_tile;
      int tile_col = col / cols_per_tile;
      int tile_index = tile_row * tiles_per_col + tile_col;

      // Add the element to the corresponding tile
      tiles[tile_index].push_back(ValueAt(row, col));
    }
  }

  return tiles;
}

std::vector<int> PrefixSumBlockMatrix::ExtractRightEdge() const {
  std::vector<int> edge(num_rows_);
  for (int row = 0; row < num_rows_; ++row) {
    edge[row] = ValueAt(row, num_cols_ - 1);
  }
  return edge;
}

std::vector<int> PrefixSumBlockMatrix::ExtractBottomEdge() const {
  std::vector<int> edge(num_cols_);
  for (int col = 0; col < num_cols_; ++col) {
    edge[col] = ValueAt(num_rows_ - 1, col);
  }
  return edge;
}

void PrefixSumBlockMatrix::AddRowwiseOffset(const std::vector<int> &offsets) {
  for (int row = 0; row < num_rows_; ++row) {
    for (int col = 0; col < num_cols_; ++col) {
      ValueAt(row, col) += offsets[row];
    }
  }
}

void PrefixSumBlockMatrix::AddColwiseOffset(const std::vector<int> &offsets) {
  for (int row = 0; row < num_rows_; ++row) {
    for (int col = 0; col < num_cols_; ++col) {
      ValueAt(row, col) += offsets[col];
    }
  }
}
