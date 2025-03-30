#include "mpi_prefix_sum/mpi_prefix_sum_solver.hpp"
#include "mpi_prefix_sum/prefix_sum_block_matrix.hpp"
#include "mpi_prefix_sum/prefix_sum_distributor.hpp"
#include "mpi_prefix_sum/mpi_cartesian_grid.hpp"

MpiPrefixSumSolver::MpiPrefixSumSolver(const MpiEnvironment& mpi, const ProgramArgs& args)
    : mpi_(mpi), args_(args) {}

void MpiPrefixSumSolver::Compute(std::vector<int>& sum_matrix) {
  MpiCartesianGrid grid(mpi_.rank(), mpi_.size());

  PrefixSumBlockMatrix matrix(args_.local_n());
  matrix.data() = sum_matrix;
  matrix.ComputeLocalPrefixSum();

  PrefixSumDistributor distributor(
      matrix,
      grid.proc_row(),
      grid.proc_col(),
      grid.grid_dim()
  );

  distributor.Distribute(grid.row_comm(), grid.col_comm());

  sum_matrix = matrix.data();
}