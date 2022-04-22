#include "mpi.h"
#include <cmath>

// sum_val for process with rank 0 will be the sum of all my_val’s
void my_prefix_sum(int local_n, int *sum_matrix)
{
    // update sum_vals to be the prefix sum
    int myrank, nprocs; // add any other variables you need

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    int p = (int)round(std::sqrt(nprocs));
}
