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
#include <string>
#include <vector>

#include "common/matrix_init.hpp"
#include "common/program_args.hpp"
#include "common/time_utils.hpp"

#include "mpi_prefix_sum/internal/mpi_cartesian_grid.hpp"
#include "mpi_prefix_sum/internal/mpi_tile_info_distributor.hpp"
#include "mpi_prefix_sum/internal/prefix_sum_block_matrix.hpp"

MpiPrefixSumSolver::MpiPrefixSumSolver(const ProgramArgs &program_args)
    : mpi_environment_(MpiEnvironment(program_args))
    , program_args_(program_args)
    , grid_(MpiCartesianGrid(
          mpi_environment_.rank(),
          program_args_.TileGridDim()[0],
          program_args_.TileGridDim()[1]
      ))
    , full_matrix_(PrefixSumBlockMatrix(0, 0))
    , assigned_matrix_(PrefixSumBlockMatrix(
          program_args_.tile_dim()[0],
          program_args_.tile_dim()[1]
      )) {

  PopulateFullMatrix();
  std::vector<std::string> time_interval_names =
      {"warmup", "total", "data_distribute", "compute", "data_gather"};
  time_intervals_.AttachIntervals(time_interval_names);

  // TODO : Error if product of num_tile_rows and num_tile_cols is not equal to
  // mpi_environment_.size()
  // TODO : Error if num_tile_rows and num_tile_cols are not divisible by
}

// void MpiPrefixSumSolver::AttachTimeInterval(std::string name) {
//   time_intervals_[name] = TimeInterval();
// }

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

  time_intervals_.RecordStart("data_distribute");
  DistributeSubMatrices();
  time_intervals_.RecordEnd("data_distribute");

  time_intervals_.RecordStart("compute");
  ComputeAndShareAssigned();
  time_intervals_.RecordEnd("compute");

  time_intervals_.RecordStart("data_gather");
  CollectSubMatrices();
  time_intervals_.RecordEnd("data_gather");
}

void MpiPrefixSumSolver::StartTimer() {
  MPI_Barrier(MPI_COMM_WORLD);
  time_intervals_.RecordStart("total");
}

void MpiPrefixSumSolver::StopTimer() { time_intervals_.RecordEnd("total"); }

void MpiPrefixSumSolver::ReportTime() const {

  auto local_start_time_s = time_intervals_.StartTime("total").count();
  auto local_end_time_s = time_intervals_.EndTime("total").count();
  auto local_elapsed_time_s = time_intervals_.ElapsedTime("total").count();

  auto local_data_distribute_time_s =
      time_intervals_.ElapsedTime("data_distribute").count();
  auto local_compute_time_s = time_intervals_.ElapsedTime("compute").count();
  auto local_data_gather_time_s =
      time_intervals_.ElapsedTime("data_gather").count();

  int rank = mpi_environment_.rank();
  int size = mpi_environment_.size();

  std::vector<double> all_start_times_s(size);
  std::vector<double> all_end_times_s(size);
  std::vector<double> all_elapsed_times_s(size);
  std::vector<double> all_data_distribute_times_s(size);
  std::vector<double> all_compute_times_s(size);
  std::vector<double> all_data_gather_times_s(size);

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
  MPI_Gather(
      &local_data_distribute_time_s,
      1,
      MPI_DOUBLE,
      all_data_distribute_times_s.data(),
      1,
      MPI_DOUBLE,
      0,
      MPI_COMM_WORLD
  );
  MPI_Gather(
      &local_compute_time_s,
      1,
      MPI_DOUBLE,
      all_compute_times_s.data(),
      1,
      MPI_DOUBLE,
      0,
      MPI_COMM_WORLD
  );
  MPI_Gather(
      &local_data_gather_time_s,
      1,
      MPI_DOUBLE,
      all_data_gather_times_s.data(),
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

    std::cout << "\n\n=== Runtime Report ===\n";
    std::cout << "Total runtime (wall clock): "
              << total_time_elapsed_time_s * 1000.0 << " ms\n";

    std::cout << "\n=== Per-Rank Timing Breakdown (ms) ===\n";
    std::cout << std::setw(8) << "Rank" << std::setw(15) << "Total"
              << std::setw(15) << "Distribute" << std::setw(15) << "Compute"
              << std::setw(15) << "Gather"
              << "\n";

    std::cout << std::string(68, '-') << "\n";

    for (int i = 0; i < size; ++i) {
      std::cout << std::setw(8) << i << std::setw(15) << std::fixed
                << std::setprecision(4) << all_elapsed_times_s[i] * 1000.0
                << std::setw(15) << all_data_distribute_times_s[i] * 1000.0
                << std::setw(15) << all_compute_times_s[i] * 1000.0
                << std::setw(15) << all_data_gather_times_s[i] * 1000.0
                << "\n";
    }

    // std::cout << "\nPer-rank execution times:\n";
    // for (int i = 0; i < size; ++i) {
    //   std::cout << "  Rank " << i << ": " << all_elapsed_times_s[i] * 1000.0
    //             << " ms\n";
    // }

    // std::cout << "\nPer-rank data distribute times:\n";
    // for (int i = 0; i < size; ++i) {
    //   std::cout << "  Rank " << i << ": "
    //             << all_data_distribute_times_s[i] * 1000.0 << " ms\n";
    // }

    // std::cout << "\nPer-rank compute times:\n";
    // for (int i = 0; i < size; ++i) {
    //   std::cout << "  Rank " << i << ": " << all_compute_times_s[i] * 1000.0
    //             << " ms\n";
    // }

    // std::cout << "\nPer-rank data gather times:\n";
    // for (int i = 0; i < size; ++i) {
    //   std::cout << "  Rank " << i << ": "
    //             << all_data_gather_times_s[i] * 1000.0 << " ms\n";
    // }

    std::cout << std::endl;
  }
}
