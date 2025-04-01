// ----------------------------------------------------------------------------
// mpi_environment.cpp
//
// Mpi environment implementation.
// ----------------------------------------------------------------------------

#include "mpi_prefix_sum/mpi_environment.hpp"

MpiEnvironment::MpiEnvironment(const ProgramArgs &program_args) {
  int argc = program_args.orig_argc();
  char **argv = program_args.orig_argv();
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &size_);
}

MpiEnvironment::~MpiEnvironment() { MPI_Finalize(); }