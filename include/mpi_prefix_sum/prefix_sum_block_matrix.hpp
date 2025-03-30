#pragma once


#include <vector>
#include <cassert>

class PrefixSumBlockMatrix {
 public:
  explicit PrefixSumBlockMatrix(int local_n);

  // Accessors
  int& ValueAt(int row, int col);
  const int& ValueAt(int row, int col) const;

  // Compute in-place local prefix sum (row-wise then col-wise)
  void ComputeLocalPrefixSum();

  // Access to raw data if needed
  std::vector<int>& data() { return data_; }
  const std::vector<int>& data() const { return data_; }

  int local_n() const { return local_n_; }

 private:
  int local_n_;
  std::vector<int> data_;
};