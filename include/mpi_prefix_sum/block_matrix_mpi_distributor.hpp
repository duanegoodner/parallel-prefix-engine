// ----------------------------------------------------------------------------
// block_matrix_mpi_distributor.hpp
//
// Prefix sum distributor definitions.
// This header is part of the prefix sum project.
// ----------------------------------------------------------------------------

#pragma once

#include "prefix_sum_block_matrix.hpp"

#include <mpi.h>

#include "mpi_prefix_sum/mpi_cartesian_grid.hpp"

// Class BlockMatrixMpiDistributor: Handles communication between processes for
// computing global prefix sums in a distributed setting.
class BlockMatrixMpiDistributor {
public:
  BlockMatrixMpiDistributor(
      PrefixSumBlockMatrix &matrix,
      const MpiCartesianGrid &grid
  );

  void ShareRightEdges(MPI_Comm row_comm);
  void ShareBottomEdges(MPI_Comm col_comm);

private:
  PrefixSumBlockMatrix &matrix_;
  const MpiCartesianGrid &grid_;
};