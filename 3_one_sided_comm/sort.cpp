#include "mpi.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include "helper.cpp"

// myItems is the input of N items, and myResult is the output with nOut items
// remember to allocate memory for myResult
void my_sort(int N, item *myItems, int *nOut, item **myResult)
{
    int rank, nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
}
