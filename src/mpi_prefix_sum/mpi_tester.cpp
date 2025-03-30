#include "common/matrix_init.hpp"
#include "mpi.h"
#include "mpi_prefix_sum/mpi_environment.hpp"
#include "mpi_prefix_sum/matrix_io.hpp"
#include "mpi_prefix_sum/mpi_prefix_sum.hpp"
#include "mpi_prefix_sum/mpi_utils.hpp"

#include <cstdlib> // for std::exit
#include <iostream>
#include <random>    // for mt19937 and uniform_int_distribution
#include <stdexcept> // for std::stoi exceptions
#include <string>
#include <vector>

int main(int argc, char *argv[]) {

  MpiEnvironment mpi(argc, argv);

  auto args = ProgramArgs::Parse(argc, argv, mpi.rank());

  auto local_mat = GenerateRandomMatrix<int>(args.local_n(), args.seed());

  if (mpi.rank() == 0)
    std::cout << "Before prefix sum:\n";
  // PrintGlobalMat(mpi.rank(), mpi.size(), args.local_n(), local_mat);

  PrintDistributedMatrix(mpi.rank(), mpi.size(), args.local_n(), local_mat);



  MyPrefixSum(args.local_n(), local_mat);

  MPI_Barrier(MPI_COMM_WORLD);
  if (mpi.rank() == 0)
    std::cout << "After prefix sum:\n";
    PrintDistributedMatrix(mpi.rank(), mpi.size(), args.local_n(), local_mat);

  return 0;
}
