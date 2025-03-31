#pragma once

#include <string>
#include <vector>

class PrefixSumSolver {
 public:
  virtual ~PrefixSumSolver() = default;
  virtual void Compute(std::vector<int>& local_matrix) = 0;
  virtual void PrintMatrix(const std::vector<int>& local_matrix,
    const std::string& header = "") const = 0;
};
