// ----------------------------------------------------------------------------
// mpi_tile_info_distributor.cpp
//
// Prefix sum distributor implementation.
// ----------------------------------------------------------------------------

#include "mpi_prefix_sum/mpi_tile_info_distributor.hpp"

#include "mpi_prefix_sum/mpi_cartesian_grid.hpp"

MpiTileInfoDistributor::MpiTileInfoDistributor(
    PrefixSumBlockMatrix &tile,
    const MpiCartesianGrid &grid
)
    : tile_(tile)
    , grid_(grid) {}

void MpiTileInfoDistributor::ShareRightEdges() {
  std::vector<int> buffer(tile_.num_rows());
  std::vector<int> accum(tile_.num_rows(), 0);

  for (int sender_col = 0; sender_col < grid_.num_cols() - 1; ++sender_col) {
    if (sender_col == grid_.proc_col()) {
      buffer = tile_.ExtractRightEdge();
    }

    MPI_Bcast(
        buffer.data(),
        tile_.num_rows(),
        MPI_INT,
        sender_col,
        grid_.row_comm()
    );
    MPI_Barrier(MPI_COMM_WORLD);

    if (grid_.proc_col() > sender_col) {
      for (int i = 0; i < tile_.num_rows(); ++i) {
        accum[i] += buffer[i];
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  tile_.AddRowwiseOffset(accum);
}

void MpiTileInfoDistributor::ShareBottomEdges() {
  std::vector<int> buffer(tile_.num_cols());
  std::vector<int> accum(tile_.num_cols(), 0);

  for (int sender_row = 0; sender_row < grid_.num_rows() - 1; ++sender_row) {
    if (sender_row == grid_.proc_row()) {
      buffer = tile_.ExtractBottomEdge();
    }

    MPI_Bcast(
        buffer.data(),
        tile_.num_cols(),
        MPI_INT,
        sender_row,
        grid_.col_comm()
    );
    MPI_Barrier(MPI_COMM_WORLD);

    if (grid_.proc_row() > sender_row) {
      for (int i = 0; i < tile_.num_cols(); ++i) {
        accum[i] += buffer[i];
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  tile_.AddColwiseOffset(accum);
}