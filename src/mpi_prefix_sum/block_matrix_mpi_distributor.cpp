// ----------------------------------------------------------------------------
// block_matrix_mpi_distributor.cpp
//
// Prefix sum distributor implementation.
// ----------------------------------------------------------------------------

#include "mpi_prefix_sum/block_matrix_mpi_distributor.hpp"

#include "mpi_prefix_sum/mpi_cartesian_grid.hpp"

BlockMatrixMpiDistributor::BlockMatrixMpiDistributor(
    PrefixSumBlockMatrix &matrix,
    const MpiCartesianGrid &grid
)
    : matrix_(matrix)
    , grid_(grid) {}

void BlockMatrixMpiDistributor::ShareRightEdges(MPI_Comm row_comm) {
  std::vector<int> buffer(matrix_.num_rows());
  std::vector<int> accum(matrix_.num_rows(), 0);

  for (int sender_col = 0; sender_col < grid_.grid_dim() - 1; ++sender_col) {
    if (sender_col == grid_.proc_col()) {
      buffer = matrix_.ExtractRightEdge();
    }

    MPI_Bcast(buffer.data(), matrix_.num_rows(), MPI_INT, sender_col, row_comm);
    MPI_Barrier(MPI_COMM_WORLD);

    if (grid_.proc_col() > sender_col) {
      for (int i = 0; i < matrix_.num_rows(); ++i) {
        accum[i] += buffer[i];
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  matrix_.AddRowwiseOffset(accum);
}

void BlockMatrixMpiDistributor::ShareBottomEdges(MPI_Comm col_comm) {
  std::vector<int> buffer(matrix_.num_cols());
  std::vector<int> accum(matrix_.num_cols(), 0);

  for (int sender_row = 0; sender_row < grid_.grid_dim() - 1; ++sender_row) {
    if (sender_row == grid_.proc_row()) {
      buffer = matrix_.ExtractBottomEdge();
    }

    MPI_Bcast(buffer.data(), matrix_.num_cols(), MPI_INT, sender_row, col_comm);
    MPI_Barrier(MPI_COMM_WORLD);

    if (grid_.proc_row() > sender_row) {
      for (int i = 0; i < matrix_.num_cols(); ++i) {
        accum[i] += buffer[i];
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  matrix_.AddColwiseOffset(accum);
}