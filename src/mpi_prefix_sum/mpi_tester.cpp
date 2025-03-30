#include "mpi.h"
#include "mpi_prefix_sum/mpi_prefix_sum.hpp"
#include "mpi_prefix_sum/mpi_utils.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>     // for std::exit
#include <stdexcept>   // for std::stoi exceptions
#include <random>      // for mt19937 and uniform_int_distribution



int main(int argc, char* argv[]) {
  
  MPI_Init(&argc, &argv);

  int myrank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  auto args = ProgramArgs::Parse(argc, argv, myrank);

  // Modern random number generation
  std::mt19937 rng(args.seed());  // Mersenne Twister RNG
  std::uniform_int_distribution<int> dist(-100, 99);

  std::vector<int> local_mat(args.local_n() * args.local_n());
  for (int& val : local_mat) {
    val = dist(rng);
  }

  if (myrank == 0)
    std::cout << "Before prefix sum:\n";
  PrintGlobalMat(myrank, nprocs, args.local_n(), local_mat);

  MyPrefixSum(args.local_n(), local_mat);

  MPI_Barrier(MPI_COMM_WORLD);
  if (myrank == 0)
    std::cout << "After prefix sum:\n";
  PrintGlobalMat(myrank, nprocs, args.local_n(), local_mat);

  MPI_Finalize();
  return 0;
}
