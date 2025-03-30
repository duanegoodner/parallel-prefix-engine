#include "mpi_prefix_sum/mpi_prefix_sum.hpp"
#include "mpi_prefix_sum/prefix_sum_block_matrix.hpp"
#include "mpi_prefix_sum/prefix_sum_distributor.hpp"
#include "mpi_prefix_sum/mpi_cartesian_grid.hpp"
#include "mpi_prefix_sum/mpi_environment.hpp"
#include "mpi_prefix_sum/mpi_utils.hpp"

#include <vector>

void MyPrefixSum(
    const MpiEnvironment& mpi,
    const ProgramArgs& args,
    std::vector<int>& sum_matrix
) {
  // Create Cartesian grid (2D, square, non-periodic)
  MpiCartesianGrid grid(mpi.rank(), mpi.size());

  // Wrap local matrix in class for local prefix operations
  PrefixSumBlockMatrix matrix(args.local_n());
  matrix.data() = sum_matrix;

  matrix.ComputeLocalPrefixSum();

  // Distribute prefix sum across rows and columns
  PrefixSumDistributor distributor(
      matrix,
      grid.proc_row(),
      grid.proc_col(),
      grid.grid_dim()
  );
  distributor.Distribute(grid.row_comm(), grid.col_comm());

  // Write result back to original vector
  sum_matrix = matrix.data();
}
