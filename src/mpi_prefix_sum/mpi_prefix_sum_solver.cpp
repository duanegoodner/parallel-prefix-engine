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

#include "common/matrix_init.hpp"
#include "common/program_args.hpp"

#include "mpi_prefix_sum/matrix_io.hpp"
#include "mpi_prefix_sum/mpi_cartesian_grid.hpp"
#include "mpi_prefix_sum/mpi_tile_info_distributor.hpp"
#include "mpi_prefix_sum/prefix_sum_block_matrix.hpp"

MpiPrefixSumSolver::MpiPrefixSumSolver(const ProgramArgs &program_args)
    : mpi_environment_(MpiEnvironment(program_args))
    , program_args_(program_args)
    , grid_(MpiCartesianGrid(
          mpi_environment_.rank(),
          program_args_.num_tile_rows(),
          program_args_.num_tile_cols()
      ))
    , full_matrix_(PrefixSumBlockMatrix(0, 0))
    , assigned_matrix_(PrefixSumBlockMatrix(
          program_args_.TileDim()[0],
          program_args_.TileDim()[1]
      )) {

  PopulateFullMatrix();

  // TODO : Error if product of num_tile_rows and num_tile_cols is not equal to
  // mpi_environment_.size()
  // TODO : Error if num_tile_rows and num_tile_cols are not divisible by
}

void MpiPrefixSumSolver::PopulateFullMatrix() {
  if (mpi_environment_.rank() == 0) {
    auto full_matrix_data = GenerateRandomMatrix<int>(
        program_args_.FullMatrixSize(),
        program_args_.seed()
    );
    // TODO build from data with correct dims(need getters in ProgramArgs)
    full_matrix_ = PrefixSumBlockMatrix(
        program_args_.full_matrix_dim()[0],
        program_args_.full_matrix_dim()[1],
        full_matrix_data
    );
  }
}

void MpiPrefixSumSolver::DistributeSubMatrices() {
  MpiTileInfoDistributor distributor(assigned_matrix_, grid_);
  distributor.DistributeFullMatrix(full_matrix_);
}

void MpiPrefixSumSolver::ComputeAndShareAssigned() {
  assigned_matrix_.ComputeLocalPrefixSum();

  MpiTileInfoDistributor distributor(assigned_matrix_, grid_);
  distributor.ShareRightEdges();
  distributor.ShareBottomEdges();
}

void MpiPrefixSumSolver::CollectSubMatrices() {
  MpiTileInfoDistributor distributor(assigned_matrix_, grid_);
  distributor.ReconstructFullMatrix(full_matrix_);
}

void MpiPrefixSumSolver::Compute(std::vector<int> &local_matrix) {

  PrefixSumBlockMatrix matrix(program_args_.local_n());
  matrix.data() = local_matrix;
  matrix.ComputeLocalPrefixSum();

  MpiTileInfoDistributor distributor(matrix, grid_);

  distributor.ShareRightEdges();
  distributor.ShareBottomEdges();
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
