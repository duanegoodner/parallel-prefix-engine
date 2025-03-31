// ----------------------------------------------------------------------------
// prefix_sum_block_matrix.cpp
//
// Prefix sum block matrix implementation.
// ----------------------------------------------------------------------------

#include "mpi_prefix_sum/prefix_sum_block_matrix.hpp"

PrefixSumBlockMatrix::PrefixSumBlockMatrix(int local_n)
    : local_n_(local_n)
    , data_(local_n * local_n) {}

int &PrefixSumBlockMatrix::ValueAt(int row, int col) {
  assert(row >= 0 && col >= 0 && row < local_n_ && col < local_n_);
  return data_[row * local_n_ + col];
}

const int &PrefixSumBlockMatrix::ValueAt(int row, int col) const {
  assert(row >= 0 && col >= 0 && row < local_n_ && col < local_n_);
  return data_[row * local_n_ + col];
}

void PrefixSumBlockMatrix::ComputeLocalPrefixSum() {
  for (int row = 0; row < local_n_; ++row) {
    for (int col = 1; col < local_n_; ++col) {
      ValueAt(row, col) += ValueAt(row, col - 1);
    }
  }

  for (int col = 0; col < local_n_; ++col) {
    for (int row = 1; row < local_n_; ++row) {
      ValueAt(row, col) += ValueAt(row - 1, col);
    }
  }
}

std::vector<int> PrefixSumBlockMatrix::ExtractRightEdge() const {
  std::vector<int> edge(local_n_);
  for (int row = 0; row < local_n_; ++row) {
    edge[row] = ValueAt(row, local_n_ - 1);
  }
  return edge;
}

std::vector<int> PrefixSumBlockMatrix::ExtractBottomEdge() const {
  std::vector<int> edge(local_n_);
  for (int col = 0; col < local_n_; ++col) {
    edge[col] = ValueAt(local_n_ - 1, col);
  }
  return edge;
}

void PrefixSumBlockMatrix::AddRowwiseOffset(const std::vector<int> &offsets) {
  for (int row = 0; row < local_n_; ++row) {
    for (int col = 0; col < local_n_; ++col) {
      ValueAt(row, col) += offsets[row];
    }
  }
}

void PrefixSumBlockMatrix::AddColwiseOffset(const std::vector<int> &offsets) {
  for (int row = 0; row < local_n_; ++row) {
    for (int col = 0; col < local_n_; ++col) {
      ValueAt(row, col) += offsets[col];
    }
  }
}
