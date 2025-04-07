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

#include "common/arg_parser.hpp"
#include "common/logger.hpp"
#include "common/matrix_init.hpp"
#include "common/prefix_sum_solver.hpp"
#include "common/program_args.hpp"
#include "common/solver_dispatch.hpp"

int main(int argc, char *argv[]) {
  auto program_args = ArgParser::Parse(argc, argv);

  Logger::SetLogLevel(program_args.log_level());

  Logger::Log(LogLevel::INFO, "Parsed options:");
  Logger::Log(
      LogLevel::INFO,
      "  rows per tile : " + std::to_string(program_args.tile_dim()[0])
  );
  Logger::Log(
      LogLevel::INFO,
      "  cols per tile : " + std::to_string(program_args.tile_dim()[1])
  );
  Logger::Log(
      LogLevel::INFO,
      "  seed    : " + std::to_string(program_args.seed())
  );
  Logger::Log(LogLevel::INFO, "  backend : " + program_args.backend());
  Logger::Log(
      LogLevel::INFO,
      "  full matrix dim : " +
          std::to_string(program_args.full_matrix_dim()[0]) + " x " +
          std::to_string(program_args.full_matrix_dim()[1])
  );


  auto solver = MakeSolver(program_args);
  auto local_mat = GenerateRandomMatrix<int>(
      program_args.full_matrix_dim()[0],
      program_args.full_matrix_dim()[1],
      program_args.seed()
  );

  solver->PrintFullMatrix("Starting matrix");
  solver->StartTimer();
  solver->Compute();
  solver->StopTimer();
  solver->PrintFullMatrix("After prefix sum computation");
  solver->ReportTime();

  return 0;
}
