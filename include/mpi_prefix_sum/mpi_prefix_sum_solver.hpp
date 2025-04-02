// ----------------------------------------------------------------------------
// mpi_prefix_sum_solver.hpp
//
// Mpi prefix sum solver definitions.
// This header is part of the prefix sum project.
// ----------------------------------------------------------------------------

#pragma once

#include <chrono>

#include "common/prefix_sum_solver.hpp"
#include "common/program_args.hpp"

#include "mpi_prefix_sum/mpi_cartesian_grid.hpp"
#include "mpi_prefix_sum/mpi_environment.hpp"
#include "mpi_prefix_sum/mpi_utils.hpp"
#include "mpi_prefix_sum/prefix_sum_block_matrix.hpp"

// Class MpiPrefixSumSolver: MPI-specific implementation of PrefixSumSolver.
// Coordinates distributed computation.
class MpiPrefixSumSolver : public PrefixSumSolver {
public:
  MpiPrefixSumSolver(const ProgramArgs &program_args);

  void PopulateFullMatrix() override;

  void Compute(std::vector<int> &local_matrix) override;

  void PrintMatrix(
      const std::vector<int> &local_matrix,
      const std::string &header = ""
  ) const override;

  void StartTimer() override;
  void StopTimer() override;
  void ReportTime() const override;

private:
  MpiEnvironment mpi_environment_;
  ProgramArgs program_args_;
  MpiCartesianGrid grid_;
  std::vector<int> full_matrix_;
  std::chrono::steady_clock::time_point start_time_, end_time_;
};

class MpiPrefixSumSolverNew : public PrefixSumSolverNew {
public:
  MpiPrefixSumSolverNew(const ProgramArgs &program_args);

  void Compute() override;

  void PrintMatrix(const std::string &header = "") const override;

  void StartTimer() override;
  void StopTimer() override;
  void ReportTime() const override;

private:
  MpiEnvironment mpi_environment_;
  ProgramArgs program_args_;
  std::chrono::steady_clock::time_point start_time_, end_time_;
};