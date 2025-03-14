#include "mpi.h"
#include "mpi_prefix_sum/mpi_prefix_sum.hpp"
#include <iostream>
#include <string>

// usage: mpirun -n <procs> ./tester <n per proc> <seed>
// seed is optional

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

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int myrank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  if (argc < 2) {
    fprintf(stderr, "Size of local matrix required\n");
    return 1;
  }

  int local_n = atoi(argv[1]);
  int *local_mat = (int *)malloc(sizeof(int) * local_n * local_n);
  // NOTE: index [i][j] is index i*local_mat+j

  if (argc > 2) {
    srand(atoi(argv[2]) + myrank);
  }

  for (int i = 0; i < local_n * local_n; i++) {
    local_mat[i] = rand() % 200 - 100;
  }

  if (myrank == 0)
    fprintf(stdout, "Before prefix sum:\n");
  print_global_mat(myrank, nprocs, local_n, local_mat);

  my_prefix_sum(local_n, local_mat);

  MPI_Barrier(MPI_COMM_WORLD);
  if (myrank == 0)
    fprintf(stdout, "After prefix sum:\n");
  print_global_mat(myrank, nprocs, local_n, local_mat);

  free(local_mat);
  MPI_Finalize();
}
