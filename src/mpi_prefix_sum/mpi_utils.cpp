#include <string>
#include "mpi.h"
#include "mpi_prefix_sum/mpi_utils.hpp"

void print_local_mat(int rank, int local_n, int *local_mat) {
    std::string output = "rank " + std::to_string(rank) + ": \n";
    for (int i = 0; i < local_n; i++) {
      for (int j = 0; j < local_n; j++) {
        output += "\t" + std::to_string(local_mat[i * local_n + j]);
      }
      output += "\n";
    }
    fprintf(stdout, "%s\n", output.c_str());
  }
  
  void print_global_mat(int rank, int procs, int local_n, int *local_mat) {
    // NOTE: sending everything to rank 0 to print is not efficient in terms of scalability
    // but ensures everything will be printed and flushed in order. This is a debugging
    // tool and not a performance level tool.
    if (rank == 0) {
      print_local_mat(rank, local_n, local_mat);
      int *recv_val = (int *)malloc(sizeof(int) * local_n * local_n);
      MPI_Status status;
      for (int i = 1; i < procs; i++) {
        // TODO: receive from each and print
        MPI_Recv(recv_val, local_n * local_n, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
        print_local_mat(i, local_n, recv_val);
      }
      free(recv_val);
    } else {
      MPI_Send(local_mat, local_n * local_n, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
  }