#pragma once

#include <vector>
#include <cassert>
#include <mpi.h>

class PrefixSumBlockMatrix {
 public:
  explicit PrefixSumBlockMatrix(int local_n);

  int& ValueAt(int row, int col);
  const int& ValueAt(int row, int col) const;

  void ComputeLocalPrefixSum();

  void BroadcastRowPrefixSums(MPI_Comm row_comm, int my_proc_col, int p);
  void BroadcastColPrefixSums(MPI_Comm col_comm, int my_proc_row, int p);

  std::vector<int>& data() { return data_; }
  const std::vector<int>& data() const { return data_; }

  int local_n() const { return local_n_; }

 private:
  int local_n_;
  std::vector<int> data_;

  std::vector<int> ExtractRightEdge() const;
  std::vector<int> ExtractBottomEdge() const;

  void AddRowwiseOffset(const std::vector<int>& offsets);
  void AddColwiseOffset(const std::vector<int>& offsets);
};
