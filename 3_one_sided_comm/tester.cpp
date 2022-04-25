#include <iostream>
#include <algorithm>
#include "sort.cpp"
#include "helper.cpp"

// NOTE: this code will be overwritten by the grader's version, so any changes you make will not persist in grading

// usage: mpirun -n <procs> ./tester <n per proc> <seed>
// seed is optional

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int myrank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc < 2) {
        fprintf(stderr, "Size of local matrix required\n");
        return 1;
    }

    int N = atoi(argv[1]);
    item *arr = (item *)malloc(sizeof(item) * N);

    if (argc > 2) {
        srand(atoi(argv[2]) + myrank);
    }

    // fill arr with random items
    for (int i = 0; i < N; i++) {
        arr[i].key = rand() % nprocs;
        arr[i].value = rand();
    }

    print_global_vals(myrank, nprocs, N, arr);

    int count;
    item* result;
    my_sort(N, arr, &count, &result);

    print_global_vals(myrank, nprocs, count, result);

    // free(arr);
    MPI_Finalize();


}
