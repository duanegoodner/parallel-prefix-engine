// ----------------------------------------------------------------------------
// mpi_prefix_sum_solver.cpp
//
// Mpi prefix sum solver implementation.
// ----------------------------------------------------------------------------

#include "mpi_prefix_sum/mpi_prefix_sum_solver.hpp"

#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

#include "common/program_args.hpp"

#include "mpi_prefix_sum/matrix_io.hpp"
#include "mpi_prefix_sum/mpi_cartesian_grid.hpp"
#include "mpi_prefix_sum/prefix_sum_block_matrix.hpp"
#include "mpi_prefix_sum/prefix_sum_distributor.hpp"

MpiPrefixSumSolver::MpiPrefixSumSolver(const ProgramArgs &program_args)
    : mpi_environment_(MpiEnvironment(program_args))
    , program_args_(program_args) {}

void MpiPrefixSumSolver::Compute(std::vector<int> &local_matrix) {
  MpiCartesianGrid grid(mpi_environment_.rank(), mpi_environment_.size());

  PrefixSumBlockMatrix matrix(program_args_.local_n());
  matrix.data() = local_matrix;
  matrix.ComputeLocalPrefixSum();

  PrefixSumDistributor
      distributor(matrix, grid.proc_row(), grid.proc_col(), grid.grid_dim());

  distributor.Distribute(grid.row_comm(), grid.col_comm());
  local_matrix = matrix.data();
}

void MpiPrefixSumSolver::PrintMatrix(
    const std::vector<int> &local_matrix,
    const std::string &header
) const {
  // Ensure all ranks have completed prior work before printing
  MPI_Barrier(MPI_COMM_WORLD);

  PrintDistributedMatrix(
      mpi_environment_.rank(),
      mpi_environment_.size(),
      program_args_.local_n(),
      local_matrix,
      header
  );
}

void MpiPrefixSumSolver::StartTimer() {
  start_time_ = std::chrono::steady_clock::now();
}

void MpiPrefixSumSolver::StopTimer() {
  end_time_ = std::chrono::steady_clock::now();
}

void MpiPrefixSumSolver::ReportTime() const {
  double local_start =
      std::chrono::duration<double>(start_time_.time_since_epoch()).count();
  double local_end =
      std::chrono::duration<double>(end_time_.time_since_epoch()).count();
  double local_elapsed =
      std::chrono::duration<double>(end_time_ - start_time_).count();

  int rank = mpi_environment_.rank();
  int size = mpi_environment_.size();

  std::vector<double> all_starts(size);
  std::vector<double> all_ends(size);
  std::vector<double> all_durations(size);

  MPI_Gather(
      &local_start,
      1,
      MPI_DOUBLE,
      all_starts.data(),
      1,
      MPI_DOUBLE,
      0,
      MPI_COMM_WORLD
  );
  MPI_Gather(
      &local_end,
      1,
      MPI_DOUBLE,
      all_ends.data(),
      1,
      MPI_DOUBLE,
      0,
      MPI_COMM_WORLD
  );
  MPI_Gather(
      &local_elapsed,
      1,
      MPI_DOUBLE,
      all_durations.data(),
      1,
      MPI_DOUBLE,
      0,
      MPI_COMM_WORLD
  );

  if (rank == 0) {
    double global_start =
        *std::min_element(all_starts.begin(), all_starts.end());
    double global_end = *std::max_element(all_ends.begin(), all_ends.end());
    double total_time = global_end - global_start;

    std::cout << "\n=== Runtime Report ===\n";
    std::cout << "Total runtime (wall clock): " << total_time * 1000
              << " ms\n";

    std::cout << "\nPer-rank execution times:\n";
    for (int i = 0; i < size; ++i) {
      std::cout << "  Rank " << i << ": " << all_durations[i] * 1000
                << " ms\n";
    }
    std::cout << std::endl;
  }
}
