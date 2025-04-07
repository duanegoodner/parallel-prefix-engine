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

#include "mpi_prefix_sum/mpi_cartesian_grid.hpp"
#include "mpi_prefix_sum/mpi_tile_info_distributor.hpp"
#include "mpi_prefix_sum/prefix_sum_block_matrix.hpp"

MpiPrefixSumSolver::MpiPrefixSumSolver(const ProgramArgs &program_args)
    : mpi_environment_(MpiEnvironment(program_args))
    , program_args_(program_args)
    , grid_(MpiCartesianGrid(
          mpi_environment_.rank(),
          program_args_.GridDim()[0],
          program_args_.GridDim()[1]
      ))
    , full_matrix_(PrefixSumBlockMatrix(0, 0))
    , assigned_matrix_(PrefixSumBlockMatrix(
          program_args_.tile_dim()[0],
          program_args_.tile_dim()[1]
      )) {

  PopulateFullMatrix();

  // TODO : Error if product of num_tile_rows and num_tile_cols is not equal to
  // mpi_environment_.size()
  // TODO : Error if num_tile_rows and num_tile_cols are not divisible by
}

void MpiPrefixSumSolver::PopulateFullMatrix() {
  if (mpi_environment_.rank() == 0) {
    auto full_matrix_data = GenerateRandomMatrix<int>(
        program_args_.full_matrix_dim()[0],
        program_args_.full_matrix_dim()[1],
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

void MpiPrefixSumSolver::Compute() {

  DistributeSubMatrices();
  // assigned_matrix_.ComputeLocalPrefixSum();
  ComputeAndShareAssigned();
  CollectSubMatrices();
}

void MpiPrefixSumSolver::StartTimer() {
  start_time_ = std::chrono::steady_clock::now();
}

void MpiPrefixSumSolver::StopTimer() {
  end_time_ = std::chrono::steady_clock::now();
}

std::chrono::duration<double> MpiPrefixSumSolver::GetElapsedTime() const {
  return end_time_ - start_time_;
}

std::chrono::duration<double> MpiPrefixSumSolver::GetStartTime() const {
  return std::chrono::duration<double>(start_time_.time_since_epoch());
}

std::chrono::duration<double> MpiPrefixSumSolver::GetEndTime() const {
  return std::chrono::duration<double>(end_time_.time_since_epoch());
}

void MpiPrefixSumSolver::ReportTime() const {
  // double local_start =
  //     std::chrono::duration<double>(start_time_.time_since_epoch()).count();
  // double local_end =
  //     std::chrono::duration<double>(end_time_.time_since_epoch()).count();
  // double local_elapsed =
  //     std::chrono::duration<double>(end_time_ - start_time_).count();

  auto local_start_time_s = GetStartTime().count();
  auto local_end_time_s = GetEndTime().count();
  auto local_elapsed_time_s = GetElapsedTime().count();

  int rank = mpi_environment_.rank();
  int size = mpi_environment_.size();

  std::vector<double> all_start_times_s(size);
  std::vector<double> all_end_times_s(size);
  std::vector<double> all_elapsed_times_s(size);

  MPI_Gather(
      &local_start_time_s,
      1,
      MPI_DOUBLE,
      all_start_times_s.data(),
      1,
      MPI_DOUBLE,
      0,
      MPI_COMM_WORLD
  );
  MPI_Gather(
      &local_end_time_s,
      1,
      MPI_DOUBLE,
      all_end_times_s.data(),
      1,
      MPI_DOUBLE,
      0,
      MPI_COMM_WORLD
  );
  MPI_Gather(
      &local_elapsed_time_s,
      1,
      MPI_DOUBLE,
      all_elapsed_times_s.data(),
      1,
      MPI_DOUBLE,
      0,
      MPI_COMM_WORLD
  );

  if (rank == 0) {
    double global_start_time_s =
        *std::min_element(all_start_times_s.begin(), all_start_times_s.end());
    double global_end_time_s =
        *std::max_element(all_end_times_s.begin(), all_end_times_s.end());
    double total_time_elapsed_time_s = global_end_time_s - global_start_time_s;

    std::cout << "\n=== Runtime Report ===\n";
    std::cout << "Total runtime (wall clock): "
              << total_time_elapsed_time_s * 1000.0 << " ms\n";

    std::cout << "\nPer-rank execution times:\n";
    for (int i = 0; i < size; ++i) {
      std::cout << "  Rank " << i << ": " << all_elapsed_times_s[i] * 1000.0
                << " ms\n";
    }
    std::cout << std::endl;
  }
}
