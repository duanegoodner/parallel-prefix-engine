// ----------------------------------------------------------------------------
// mpi_prefix_sum_solver.hpp
//
// Mpi prefix sum solver definitions.
// This header is part of the prefix sum project.
// ----------------------------------------------------------------------------

#pragma once

#include <chrono>
#include <string>
#include <unordered_map>

#include "common/prefix_sum_solver.hpp"
#include "common/program_args.hpp"
#include "common/time_utils.hpp"

#include "mpi_prefix_sum/internal/mpi_cartesian_grid.hpp"
#include "mpi_prefix_sum/internal/mpi_environment.hpp"
#include "mpi_prefix_sum/internal/mpi_utils.hpp"
#include "mpi_prefix_sum/internal/prefix_sum_block_matrix.hpp"

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
  void PrintLowerRightElement(std::string title = "") override {
    if (mpi_environment_.rank() == 0) {
      std::cout << title << std::endl;
      std::cout << full_matrix_.ValueAt(
          full_matrix_.num_rows() - 1,
          full_matrix_.num_cols() - 1
      );
    }
  };
  void PrintAssignedMatrix() { assigned_matrix_.Print(); }
  void StartTimer() override;
  void StopTimer() override;
  void ReportTime() const override;

private:
  MpiEnvironment mpi_environment_;
  ProgramArgs program_args_;
  MpiCartesianGrid grid_;
  PrefixSumBlockMatrix full_matrix_;
  PrefixSumBlockMatrix assigned_matrix_;
  TimeIntervals time_intervals_;
};
