#pragma once

/**
 * @file cuda_prefix_sum_solver.hpp
 * @brief CUDA implementation of the 2D prefix sum solver interface.
 */

#include <string>
#include <vector>

#include "common/prefix_sum_solver.hpp"
#include "common/program_args.hpp"

/**
 * @class CudaPrefixSumSolver
 * @brief Performs 2D prefix sum using CUDA (GPU backend).
 */
class CudaPrefixSumSolver : public PrefixSumSolver {
public:
  /**
   * @brief Construct a new CudaPrefixSumSolver using parsed CLI args.
   * @param argc Command-line argument count.
   * @param argv Command-line argument values.
   */
  CudaPrefixSumSolver(int argc, char *argv[]);

  void Compute(std::vector<int> &local_matrix) override;

  void PrintMatrix(
      const std::vector<int> &local_matrix,
      const std::string &header
  ) const override;

  void StartTimer() override;
  void StopTimer() override;
  void ReportTime() const override;

private:
  ProgramArgs args_;
  float execution_time_ms_ = 0.0f;
};
