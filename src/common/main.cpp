// -----------------------------------------------------------------------------
// main.cpp
//
// Entry point for the 2D prefix sum program. Uses a backend-specific solver
// (e.g., MPI) to compute the prefix sum of a distributed or local matrix.
//
// This program supports backend selection via command-line arguments, using
// CLI11 for argument parsing.
//
// USAGE (runnng from project root):
//   mpirun -n <num_procs> ./build/bin/prefix_sum <local_n> [--seed <int>]
//   [--backend <backend_name>]
//
// ARGUMENTS:
//   <local_n>             Size of each local matrix block (NxN)
//   --seed <int>          Optional seed for reproducible random generation
//   --backend <string>    Backend to use (e.g., "mpi"). Currently only "mpi"
//   is supported.
//
// EXAMPLES  (runnng from project root):
//   mpirun -n 4 ./build/bin/prefix_sum 2
//   mpirun -n 4 ./build/bin/prefix_sum 4 --seed 42 --backend mpi
//
// OUTPUT:
//   Printed matrices before and after prefix sum (rank-ordered output from
//   rank 0).
//
// -----------------------------------------------------------------------------

#include <iostream>
#include <memory>
#include <vector>

#include "common/logger.hpp"
#include "common/matrix_init.hpp"
#include "common/prefix_sum_solver.hpp"
#include "common/program_args.hpp"

int main(int argc, char *argv[]) {
  ProgramArgs program_args = ProgramArgs::Parse(argc, argv);
  // Logger::SetVerbose(args.verbose()); // 👈 enable debug messages if
  // requested

  // Logger::Log(
  //     LogLevel::INFO,
  //     "Creating solver for backend: " + args.backend()
  // );

  std::cout << "argc " << argc << std::endl;
  for (auto idx = 0; idx < argc; ++idx) {
    std::cout << "argv[" << idx << "] = " << argv[idx] << std::endl;
  }

  if (program_args.verbose()) {
    std::cout << "Parsed options:\n"
              << "  local_n : " << program_args.local_n() << "\n"
              << "  seed    : " << program_args.seed() << "\n"
              << "  backend : " << program_args.backend() << "\n"
              << std::endl;
  }

  auto solver = program_args.MakeSolver();
  auto local_mat = GenerateRandomMatrix<int>(program_args.local_n(), program_args.seed());

  // Logger::Log(LogLevel::DEBUG, "Random matrix initialized.");

  solver->PrintMatrix(local_mat, "Before prefix sum:");
  solver->StartTimer();
  solver->Compute(local_mat);
  solver->StopTimer();
  solver->ReportTime();
  solver->PrintMatrix(local_mat, "After prefix sum:");

  return 0;
}
