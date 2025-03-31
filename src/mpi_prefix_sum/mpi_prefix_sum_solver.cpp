#include "common/program_args.hpp"
#include "mpi_prefix_sum/mpi_prefix_sum_solver.hpp"
#include "mpi_prefix_sum/matrix_io.hpp"
#include "mpi_prefix_sum/mpi_cartesian_grid.hpp"
#include "mpi_prefix_sum/prefix_sum_block_matrix.hpp"
#include "mpi_prefix_sum/prefix_sum_distributor.hpp"

MpiPrefixSumSolver::MpiPrefixSumSolver(int argc, char* argv[])
    : mpi_(argc, argv),
      args_(ProgramArgs::ParseForMPI(argc, argv, mpi_.rank())) {}

void MpiPrefixSumSolver::Compute(std::vector<int> &local_matrix) {
  MpiCartesianGrid grid(mpi_.rank(), mpi_.size());

  PrefixSumBlockMatrix matrix(args_.local_n());
  matrix.data() = local_matrix;
  matrix.ComputeLocalPrefixSum();

  PrefixSumDistributor
      distributor(matrix, grid.proc_row(), grid.proc_col(), grid.grid_dim());

  distributor.Distribute(grid.row_comm(), grid.col_comm());
  local_matrix = matrix.data();
}

void MpiPrefixSumSolver::PrintMatrix(
    const std::vector<int> &local_matrix,
    const std::string &header
) const {
  // Ensure all ranks have completed prior work before printing
  MPI_Barrier(MPI_COMM_WORLD);

  PrintDistributedMatrix(
      mpi_.rank(),
      mpi_.size(),
      args_.local_n(),
      local_matrix,
      header
  );
}
