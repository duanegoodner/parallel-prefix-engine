// -----------------------------------------------------------------------------
// mpi_tester.cpp
//
// Entry point for running a distributed 2D prefix sum using MPI.
//
// USAGE:
//   mpirun -n <num_procs> ./prefix_sum_mpi <local_n> [seed]
//
// ARGUMENTS:
//   <num_procs>   Number of MPI processes (must be a perfect square, e.g., 4, 9, 16)
//   <local_n>     Size of each block's local matrix (matrix is local_n x local_n)
//   [seed]        Optional random seed; if provided, will be offset per rank
//
// EXAMPLE:
//   mpirun -n 4 ./prefix_sum_mpi 2 1234
//
//   This launches 4 MPI processes (arranged as 2x2 grid),
//   each generating a 2x2 matrix filled with random values,
//   and performs a distributed prefix sum across the global 4x4 matrix.
//
// OUTPUT:
//   Printed before/after matrices gathered and shown in rank order by process 0.
//
// -----------------------------------------------------------------------------


#include "mpi.h"
#include "common/matrix_init.hpp"
#include "mpi_prefix_sum/matrix_io.hpp"
#include "mpi_prefix_sum/mpi_environment.hpp"
#include "mpi_prefix_sum/mpi_prefix_sum.hpp"
#include "mpi_prefix_sum/mpi_utils.hpp"

#include <vector>
#include <string>

int main(int argc, char *argv[]) {
  // Initialize MPI and get rank/size info
  MpiEnvironment mpi(argc, argv);

  // Parse command-line arguments
  auto args = ProgramArgs::Parse(argc, argv, mpi.rank());

  // Generate local matrix filled with random integers
  auto local_mat = GenerateRandomMatrix<int>(args.local_n(), args.seed());

  // Print matrix before prefix sum computation
  PrintDistributedMatrix(
      mpi.rank(),
      mpi.size(),
      args.local_n(),
      local_mat,
      "Before prefix sum:"
  );

  // Perform distributed 2D prefix sum
  MyPrefixSum(args.local_n(), local_mat);

  // Synchronize before printing results
  MPI_Barrier(MPI_COMM_WORLD);

  // Print matrix after prefix sum computation
  PrintDistributedMatrix(
      mpi.rank(),
      mpi.size(),
      args.local_n(),
      local_mat,
      "After prefix sum:"
  );

  return 0;
}
