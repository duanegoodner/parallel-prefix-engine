// cuda_prefix_sum_solver.hpp
//
// Declares the CudaPrefixSumSolver class for performing distributed prefix
// sum using CUDA. Implements the PrefixSumSolver interface.

#pragma once

#include <chrono>
#include <string>
#include <vector>

#include "common/prefix_sum_solver.hpp"
#include "common/program_args.hpp"

class CudaPrefixSumSolver : public PrefixSumSolver {
public:
  explicit CudaPrefixSumSolver(const ProgramArgs &program_args);

  void Compute(std::vector<int> &local_matrix) override;
  void PrintMatrix(
      const std::vector<int> &local_matrix,
      const std::string &header
  ) const override;

  const ProgramArgs &program_args() const;

  void StartTimer() override;
  void StopTimer() override;
  void ReportTime() const override;

private:
  ProgramArgs program_args_;
  std::chrono::steady_clock::time_point start_time_;
  std::chrono::steady_clock::time_point stop_time_;
};
