#include "mpi_prefix_sum/mpi_utils.hpp"
#include "mpi.h"
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <string>

ProgramArgs ProgramArgs::Parse(int argc, char* argv[], int rank) {
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
  } catch (const std::invalid_argument&) {
    if (rank == 0) std::cerr << "Invalid numeric argument.\n";
    std::exit(1);
  } catch (const std::out_of_range&) {
    if (rank == 0) std::cerr << "Argument out of range.\n";
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

void PrintGlobalMat(
    int rank,
    int procs,
    int local_n,
    const std::vector<int> &local_mat
) {
  // NOTE: sending everything to rank 0 to print is not efficient in terms of scalability
  // but ensures everything will be printed and flushed in order. This is a debugging
  // tool and not a performance level tool.
  if (rank == 0) {
    PrintLocalMat(rank, local_n, local_mat);
    // int *recv_val = (int *)malloc(sizeof(int) * local_n * local_n);
    std::vector<int> recv_val(local_n * local_n);
    MPI_Status status;
    for (int i = 1; i < procs; i++) {
      // TODO: receive from each and print
      MPI_Recv(recv_val.data(), local_n * local_n, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
      PrintLocalMat(i, local_n, recv_val);
    }
    // free(recv_val);
  } else {
    MPI_Send(local_mat.data(), local_n * local_n, MPI_INT, 0, 0, MPI_COMM_WORLD);
  }
}