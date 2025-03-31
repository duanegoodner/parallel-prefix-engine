// -----------------------------------------------------------------------------
// main.cpp
//
// Entry point for distributed 2D prefix sum computation. This program supports
// multiple parallel backends (currently MPI) and cleanly separates algorithm
// logic from CLI and orchestration via a PrefixSumSolver interface.
//
// This file is backend-agnostic. It parses command-line arguments, dispatches
// to the appropriate solver implementation, and manages input/output for
// matrix generation and result visualization.
//
// USAGE:
//   mpirun -n <num_procs> ./prefix_sum_mpi <local_n> [seed] [--backend=mpi]
//
// ARGUMENTS:
//   <local_n>        Size of each rank's local matrix (NxN block)
//   [seed]           Optional seed for reproducible random values
//   [--backend=...]  Parallel backend to use (default: mpi)
//
// EXAMPLES:
//   mpirun -n 4 ./prefix_sum_mpi 2 --backend=mpi
//     → 4 MPI processes arranged as 2x2 grid, each with a 2x2 matrix block.
//
// BACKENDS:
//   - mpi    Uses MPI for distributed prefix sum (default)
//   - cuda   (Planned) GPU implementation using CUDA
//
// OUTPUT:
//   Matrix state before and after prefix sum, printed in rank order via rank 0.
//
// NOTES:
//   - Number of processes must be a perfect square (e.g., 4, 9, 16)
//   - Backend implementations must inherit from PrefixSumSolver interface
//
// -----------------------------------------------------------------------------


#include "common/matrix_init.hpp"
#include "common/prefix_sum_solver.hpp"
#include "common/program_args.hpp"
#include "mpi_prefix_sum/mpi_prefix_sum_solver.hpp"

#include <iostream>
#include <memory>
#include <vector>

int main(int argc, char* argv[]) {
  ProgramArgs args = ProgramArgs::Parse(argc, argv);

  std::unique_ptr<PrefixSumSolver> solver;

  if (args.backend() == "mpi") {
    solver = std::make_unique<MpiPrefixSumSolver>(argc, argv);
  } else {
    std::cerr << "Unsupported backend: " << args.backend() << "\n";
    ProgramArgs::PrintUsage(std::cerr);
    return 1;
  }

  auto local_mat = GenerateRandomMatrix<int>(args.local_n(), args.seed());

  solver->PrintMatrix(local_mat, "Before prefix sum:");
  solver->Compute(local_mat);
  solver->PrintMatrix(local_mat, "After prefix sum:");

  return 0;
}
