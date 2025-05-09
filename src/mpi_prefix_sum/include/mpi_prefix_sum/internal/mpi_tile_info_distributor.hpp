// ----------------------------------------------------------------------------
// mpi_tile_info_distributor.hpp
//
// Prefix sum distributor definitions.
// This header is part of the prefix sum project.
// ----------------------------------------------------------------------------

#pragma once

#include "prefix_sum_block_matrix.hpp"

#include <mpi.h>

#include "mpi_prefix_sum/internal/mpi_cartesian_grid.hpp"

// Class MpiTileInfoDistributor: Handles communication between processes for
// computing global prefix sums in a distributed setting.
class MpiTileInfoDistributor {
public:
  MpiTileInfoDistributor(
      PrefixSumBlockMatrix &tile,
      const MpiCartesianGrid &grid
  );

  void DistributeFullMatrix(const PrefixSumBlockMatrix &full_matrix);

  void ReconstructFullMatrix(PrefixSumBlockMatrix &full_matrix);

  void ShareRightEdges();
  void ShareBottomEdges();

private:
  PrefixSumBlockMatrix &tile_;
  const MpiCartesianGrid &grid_;
};