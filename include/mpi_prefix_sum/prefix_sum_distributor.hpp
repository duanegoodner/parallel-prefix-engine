// ----------------------------------------------------------------------------
// prefix_sum_distributor.hpp
//
// Prefix sum distributor definitions.
// This header is part of the prefix sum project.
// ----------------------------------------------------------------------------

#pragma once

#include "prefix_sum_block_matrix.hpp"

#include <mpi.h>

#include "mpi_prefix_sum/mpi_cartesian_grid.hpp"

// Class PrefixSumDistributor: Handles communication between processes for
// computing global prefix sums in a distributed setting.
class PrefixSumDistributor {
public:
  PrefixSumDistributor(
      PrefixSumBlockMatrix &matrix,
      const MpiCartesianGrid &grid,
      // int proc_row,
      // int proc_col,
      int p
  );

  void Distribute(MPI_Comm comm_row, MPI_Comm comm_col);

private:
  void BroadcastRowPrefixSums(MPI_Comm row_comm);
  void BroadcastColPrefixSums(MPI_Comm col_comm);

  PrefixSumBlockMatrix &matrix_;
  const MpiCartesianGrid &grid_;
  // int proc_row_;
  // int proc_col_;
  int p_;
};