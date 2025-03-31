#pragma once

#include "common/prefix_sum_solver.hpp"
#include "common/program_args.hpp"
#include "mpi_prefix_sum/mpi_environment.hpp"
#include "mpi_prefix_sum/mpi_utils.hpp"
#include <chrono>

class MpiPrefixSumSolver : public PrefixSumSolver {
public:
  MpiPrefixSumSolver(int argc, char *argv[]);

  void Compute(std::vector<int> &local_matrix) override;

  void PrintMatrix(
      const std::vector<int> &local_matrix,
      const std::string &header = ""
  ) const override;

  const ProgramArgs &args() const { return args_; }
  const MpiEnvironment &mpi() const { return mpi_; }

  void StartTimer() override;
  void StopTimer() override;
  void ReportTime() const override;

private:
  MpiEnvironment mpi_;
  ProgramArgs args_;
  std::chrono::steady_clock::time_point start_time_, end_time_;
};
