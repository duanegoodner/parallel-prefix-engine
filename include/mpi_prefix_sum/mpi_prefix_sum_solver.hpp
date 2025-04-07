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

  void DistributeSubMatrices();

  void ComputeAndShareAssigned();

  void CollectSubMatrices();

  void Compute() override;

  int Rank() const { return mpi_environment_.rank(); }

  void PrintFullMatrix(std::string title = "") override {
    
    if (mpi_environment_.rank() == 0) {
      std::cout << title << std::endl;
      full_matrix_.Print();
    }
  }

  void PrintAssignedMatrix() { assigned_matrix_.Print(); }

  void StartTimer() override;
  void StopTimer() override;
  std::chrono::duration<double> GetElapsedTime() const override;
  std::chrono::duration<double> GetStartTime() const override;
  std::chrono::duration<double> GetEndTime() const override;

  void ReportTime() const override;

private:
  MpiEnvironment mpi_environment_;
  ProgramArgs program_args_;
  MpiCartesianGrid grid_;
  PrefixSumBlockMatrix full_matrix_;
  PrefixSumBlockMatrix assigned_matrix_;
  std::chrono::steady_clock::time_point start_time_, end_time_;
};
