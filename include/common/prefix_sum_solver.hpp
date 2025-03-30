#pragma once

#include <vector>

class PrefixSumSolver {
 public:
  virtual ~PrefixSumSolver() = default;
  virtual void Compute(std::vector<int>& local_matrix) = 0;
};
