#include "mpi_prefix_sum/mpi_utils.hpp"
#include "mpi.h"
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

ProgramArgs ProgramArgs::Parse(int argc, char* const argv[], int rank) {
  ProgramArgs args;

  if (argc < 2) {
    if (rank == 0) {
      std::cerr << "Usage: " << argv[0] << " <local_n> [seed]\n";
    }
    std::exit(1);
  }

  try {
    args.set_local_n(std::stoi(argv[1]));

    if (argc > 2) {
      args.set_seed(std::stoi(argv[2]) + rank);
    }
  } catch (const std::invalid_argument &) {
    if (rank == 0)
      std::cerr << "Invalid numeric argument.\n";
    std::exit(1);
  } catch (const std::out_of_range &) {
    if (rank == 0)
      std::cerr << "Argument out of range.\n";
    std::exit(1);
  }

  return args;
}

void PrintLocalMat(int rank, int local_n, const std::vector<int> &local_mat) {
  std::string output = "rank " + std::to_string(rank) + ": \n";
  for (int i = 0; i < local_n; i++) {
    for (int j = 0; j < local_n; j++) {
      output += "\t" + std::to_string(local_mat[i * local_n + j]);
    }
    output += "\n";
  }
  fprintf(stdout, "%s\n", output.c_str());
}
