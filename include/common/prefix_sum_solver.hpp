// ----------------------------------------------------------------------------
// prefix_sum_solver.hpp
//
// Prefix sum solver definitions.
// This header is part of the prefix sum project.
// ----------------------------------------------------------------------------

#pragma once

#include <string>
#include <vector>

// Class PrefixSumSolver: Abstract interface for performing 2D prefix sums
// using different backends (MPI, CUDA, etc).
class PrefixSumSolver {
public:
  virtual ~PrefixSumSolver() = default;

  virtual void PopulateFullMatrix() = 0;

  virtual void Compute() = 0;

  virtual void PrintFullMatrix() = 0;

  // ⏱️ Time tracking (backend-specific implementations)
  virtual void StartTimer() = 0;
  virtual void StopTimer() = 0;
  virtual void ReportTime() const = 0;
};
