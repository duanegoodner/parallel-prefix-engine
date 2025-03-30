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

  auto args = get_args(argc, argv, myrank);
  auto local_n = args.local_n;
  auto seed = args.seed;

  // int local_n = 0;
  // int seed = 1234;  // Default seed if none provided

  // try {
  //   if (argc < 2) {
  //     if (myrank == 0) {
  //       std::cerr << "Usage: " << argv[0] << " <local_n> [seed]\n";
  //     }
  //     MPI_Finalize();
  //     return 1;
  //   }

  //   local_n = std::stoi(argv[1]);

  //   if (argc > 2) {
  //     seed = std::stoi(argv[2]) + myrank;
  //   }
  // } catch (const std::invalid_argument& e) {
  //   if (myrank == 0) {
  //     std::cerr << "Error: Invalid numeric argument.\n";
  //   }
  //   MPI_Finalize();
  //   return 1;
  // } catch (const std::out_of_range& e) {
  //   if (myrank == 0) {
  //     std::cerr << "Error: Argument out of range.\n";
  //   }
  //   MPI_Finalize();
  //   return 1;
  // }

  // Modern random number generation
  std::mt19937 rng(seed);  // Mersenne Twister RNG
  std::uniform_int_distribution<int> dist(-100, 99);

  std::vector<int> local_mat(local_n * local_n);
  for (int& val : local_mat) {
    val = dist(rng);
  }

  if (myrank == 0)
    std::cout << "Before prefix sum:\n";
  print_global_mat(myrank, nprocs, local_n, local_mat);

  my_prefix_sum(local_n, local_mat);

  MPI_Barrier(MPI_COMM_WORLD);
  if (myrank == 0)
    std::cout << "After prefix sum:\n";
  print_global_mat(myrank, nprocs, local_n, local_mat);

  MPI_Finalize();
  return 0;
}
