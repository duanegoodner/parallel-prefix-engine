// ----------------------------------------------------------------------------
// mpi_utils.cpp
//
// Mpi utils implementation.
// ----------------------------------------------------------------------------

#include "mpi_prefix_sum/mpi_utils.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "mpi.h"

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
