#include "mpi_prefix_sum/mpi_prefix_sum.hpp"
#include "mpi_prefix_sum/prefix_sum_block_matrix.hpp"
#include "mpi_prefix_sum/prefix_sum_distributor.hpp"
#include <cmath>
#include <mpi.h>
#include <vector>

void MyPrefixSum(int local_n, std::vector<int>& sum_matrix) {
  int myrank, nprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  int p = static_cast<int>(std::round(std::sqrt(nprocs)));

  int dimsize[2] = {p, p};
  int periodic[2] = {0, 0};
  MPI_Comm comm_2d;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dimsize, periodic, 0, &comm_2d);

  int my_proc_coords[2];
  MPI_Cart_coords(comm_2d, myrank, 2, my_proc_coords);
  int my_proc_row = my_proc_coords[0];
  int my_proc_col = my_proc_coords[1];

  MPI_Comm comm_row, comm_col;
  MPI_Comm_split(comm_2d, my_proc_row, my_proc_col, &comm_row);
  MPI_Comm_split(comm_2d, my_proc_col, my_proc_row, &comm_col);

  PrefixSumBlockMatrix matrix(local_n);
  matrix.data() = sum_matrix;

  matrix.ComputeLocalPrefixSum();

  PrefixSumDistributor distributor(matrix, my_proc_row, my_proc_col, p);
  distributor.Distribute(comm_row, comm_col);

  // matrix.BroadcastRowPrefixSums(comm_row, my_proc_col, p);
  // matrix.BroadcastColPrefixSums(comm_col, my_proc_row, p);

  sum_matrix = matrix.data();
}