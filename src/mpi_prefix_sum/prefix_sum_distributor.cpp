// ----------------------------------------------------------------------------
// prefix_sum_distributor.cpp
//
// Prefix sum distributor implementation.
// ----------------------------------------------------------------------------

#include "mpi_prefix_sum/prefix_sum_distributor.hpp"

#include "mpi_prefix_sum/mpi_cartesian_grid.hpp"

PrefixSumDistributor::PrefixSumDistributor(
    PrefixSumBlockMatrix &matrix,
    const MpiCartesianGrid &grid
    // int proc_row,
    // int proc_col,
    // int p
)
    : matrix_(matrix)
    , grid_(grid) {}
    // , proc_row_(proc_row)
    // , proc_col_(proc_col)
    // , p_(p) {}

void PrefixSumDistributor::Distribute(MPI_Comm comm_row, MPI_Comm comm_col) {
  BroadcastRowPrefixSums(comm_row);
  BroadcastColPrefixSums(comm_col);
}

void PrefixSumDistributor::BroadcastRowPrefixSums(MPI_Comm row_comm) {
  std::vector<int> buffer(matrix_.local_n());
  std::vector<int> accum(matrix_.local_n(), 0);

  for (int sender_col = 0; sender_col < grid_.grid_dim() - 1; ++sender_col) {
    if (sender_col == grid_.proc_col()) {
      buffer = matrix_.ExtractRightEdge();
    }

    MPI_Bcast(buffer.data(), matrix_.local_n(), MPI_INT, sender_col, row_comm);
    MPI_Barrier(MPI_COMM_WORLD);

    if (grid_.proc_col() > sender_col) {
      for (int i = 0; i < matrix_.local_n(); ++i) {
        accum[i] += buffer[i];
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  matrix_.AddRowwiseOffset(accum);
}

void PrefixSumDistributor::BroadcastColPrefixSums(MPI_Comm col_comm) {
  std::vector<int> buffer(matrix_.local_n());
  std::vector<int> accum(matrix_.local_n(), 0);

  for (int sender_row = 0; sender_row < grid_.grid_dim() - 1; ++sender_row) {
    if (sender_row == grid_.proc_row()) {
      buffer = matrix_.ExtractBottomEdge();
    }

    MPI_Bcast(buffer.data(), matrix_.local_n(), MPI_INT, sender_row, col_comm);
    MPI_Barrier(MPI_COMM_WORLD);

    if (grid_.proc_row() > sender_row) {
      for (int i = 0; i < matrix_.local_n(); ++i) {
        accum[i] += buffer[i];
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  matrix_.AddColwiseOffset(accum);
}