// -----------------------------------------------------------------------------
// main.cpp
//
// Entry point for distributed 2D prefix sum computation.
// Backend-agnostic: selects implementation via --backend flag.
//
// USAGE:
//   mpirun -n <num_procs> ./prefix_sum_mpi <local_n> [seed] [--backend=mpi]
//
// ARGUMENTS:
//   <local_n>        Size of each process's local submatrix (NxN block)
//   [seed]           Optional seed for matrix generation
//   [--backend=...]  Select backend: mpi (default), cuda (future)
//
// EXAMPLES:
//   mpirun -n 4 ./prefix_sum_mpi 2 --backend=mpi
//
//   This runs a 2x2 process grid with each process using a 2x2 matrix.
//   A global 4x4 prefix sum will be computed using the MPI backend.
//
// OUTPUT:
//   Matrices printed in process rank order by rank 0 before and after computation.
//
// NOTES:
//   - Number of processes must form a perfect square (e.g., 4, 9, 16)
//   - This tool assumes a block-cyclic data layout across ranks
// -----------------------------------------------------------------------------


#include "common/matrix_init.hpp"
#include "common/prefix_sum_solver.hpp"
#include "common/program_args.hpp"
#include "mpi_prefix_sum/mpi_prefix_sum_solver.hpp"

#include <iostream>
#include <memory>
#include <vector>

int main(int argc, char* argv[]) {
  // Parse arguments (independent of backend)
  ProgramArgs args = ProgramArgs::Parse(argc, argv);

  // Dynamically choose backend
  std::unique_ptr<PrefixSumSolver> solver;

  if (args.backend() == "mpi") {
    solver = std::make_unique<MpiPrefixSumSolver>(argc, argv);
  } else {
    std::cerr << "Unsupported backend: " << args.backend() << "\n";
    ProgramArgs::PrintUsage(std::cerr);
    return 1;
  }

  // Generate input matrix
  auto local_mat = GenerateRandomMatrix<int>(args.local_n(), args.seed());

  solver->PrintMatrix(local_mat, "Before prefix sum:");
  solver->Compute(local_mat);
  solver->PrintMatrix(local_mat, "After prefix sum:");

  return 0;
}
