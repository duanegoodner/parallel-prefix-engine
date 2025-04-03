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

void MpiTileInfoDistributor::DistributeFullMatrix(
    const PrefixSumBlockMatrix &full_matrix
) {

  if (grid_.rank() == 0) {
    std::cout << "Rank 0 will subdivide and distribute the following matrix:"
              << std::endl;
    full_matrix.Print();
    std::cout << std::endl;
  }

  if (grid_.rank() == 0) {

    auto sub_matrices =
        full_matrix.SubDivide(grid_.num_rows(), grid_.num_cols());
    for (auto &sub_matrix : sub_matrices) {
      auto block_sub_matrix = PrefixSumBlockMatrix(
          tile_.num_rows(),
          tile_.num_cols(),
          sub_matrix.second
      );

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
  // MPI_Barrier(MPI_COMM_WORLD);
  MPI_Recv(
      tile_.data().data(),
      tile_.data().size(),
      MPI_INT,
      0,
      0,
      grid_.cart_comm(),
      MPI_STATUS_IGNORE
  );

  std::cout << "Received by rank " << grid_.rank() << std::endl;
  tile_.Print();
  std::cout << std::endl;
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