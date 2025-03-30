#include "mpi_prefix_sum/prefix_sum_block_matrix.hpp"

PrefixSumBlockMatrix::PrefixSumBlockMatrix(int local_n)
    : local_n_(local_n), data_(local_n * local_n) {}

int& PrefixSumBlockMatrix::ValueAt(int row, int col) {
  assert(row >= 0 && col >= 0 && row < local_n_ && col < local_n_);
  return data_[row * local_n_ + col];
}

const int& PrefixSumBlockMatrix::ValueAt(int row, int col) const {
  assert(row >= 0 && col >= 0 && row < local_n_ && col < local_n_);
  return data_[row * local_n_ + col];
}

void PrefixSumBlockMatrix::ComputeLocalPrefixSum() {
  // Row-wise prefix sum
  for (int row = 0; row < local_n_; ++row) {
    for (int col = 1; col < local_n_; ++col) {
      ValueAt(row, col) += ValueAt(row, col - 1);
    }
  }

  // Column-wise prefix sum
  for (int col = 0; col < local_n_; ++col) {
    for (int row = 1; row < local_n_; ++row) {
      ValueAt(row, col) += ValueAt(row - 1, col);
    }
  }
}
