// ----------------------------------------------------------------------------
// mpi_tile_info_distributor.cpp
//
// Prefix sum distributor implementation.
// ----------------------------------------------------------------------------

#include "mpi_prefix_sum/internal/mpi_tile_info_distributor.hpp"

#include "mpi_prefix_sum/internal/mpi_cartesian_grid.hpp"

MpiTileInfoDistributor::MpiTileInfoDistributor(
    PrefixSumBlockMatrix &tile,
    const MpiCartesianGrid &grid
)
    : tile_(tile)
    , grid_(grid) {}

void MpiTileInfoDistributor::DistributeFullMatrix(
    const PrefixSumBlockMatrix &full_matrix
) {

  if (grid_.rank() == 0) {

    auto sub_matrices =
        full_matrix.SubDivide(grid_.num_rows(), grid_.num_cols());

    for (auto &sub_matrix : sub_matrices) {

      if (sub_matrix.first == 0) {
        std::copy(
            sub_matrix.second.begin(),
            sub_matrix.second.end(),
            tile_.data().begin()
        );
      } else {

        MPI_Send(
            sub_matrix.second.data(),
            sub_matrix.second.size(),
            MPI_INT,
            sub_matrix.first,
            0,
            grid_.cart_comm()
        );
      }
    }
  }
  // MPI_Barrier(MPI_COMM_WORLD);

  if (grid_.rank() != 0) {
    MPI_Recv(
        tile_.data().data(),
        tile_.data().size(),
        MPI_INT,
        0,
        0,
        grid_.cart_comm(),
        MPI_STATUS_IGNORE
    );
  }
}

void MpiTileInfoDistributor::ReconstructFullMatrix(
    PrefixSumBlockMatrix &full_matrix
) {

  // All ranks (except rank 0) send data to rank 0
  if (grid_.rank() != 0) {
    MPI_Send(
        tile_.data().data(),
        tile_.data().size(),
        MPI_INT,
        0,
        0,
        grid_.cart_comm()
    );
  }

  // Rank 0 receives all tile data into a map
  if (grid_.rank() == 0) {
    std::unordered_map<int, PrefixSumBlockMatrix> tile_data;

    // rank 0 copies its own data (no send/receive)
    for (auto idx = 0; idx < grid_.size(); ++idx) {
      tile_data[idx] =
          PrefixSumBlockMatrix(tile_.num_rows(), tile_.num_cols());
    }

    // rank 0 receives data sent by all other ranks
    for (auto idx = 0; idx < tile_.data().size(); ++idx) {
      tile_data[0].data()[idx] = tile_.data()[idx];
    }

    for (auto idx = 1; idx < grid_.size(); ++idx) {
      MPI_Recv(
          tile_data[idx].data().data(),
          tile_data[idx].data().size(),
          MPI_INT,
          idx,
          0,
          grid_.cart_comm(),
          MPI_STATUS_IGNORE
      );
    }

    PrefixSumBlockMatrix::Combine(
        tile_data,
        grid_.num_rows(),
        grid_.num_cols(),
        full_matrix
    );
  }
}

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