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

    int my_counts[nprocs];
    for (int count_index = 0; count_index < nprocs; count_index++) {
        my_counts[count_index] = 0;
    }

    for (int my_item_index = 0; my_item_index < N; my_item_index++) {
        my_counts[myItems[my_item_index].key] += 1;
    }

    std::cout << "Process " << rank << "'s key_value counts are "
        << my_counts[0] << ", "
        << my_counts[1] << ", "
        << my_counts[2]
        << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

    int my_prefix_counts[nprocs];
    for (int prefix_count_index = 0; prefix_count_index < nprocs; prefix_count_index++) {
        my_prefix_counts[prefix_count_index] = 0;
    }

    int total_items_for_me;
    for (int receiving_rank = 0; receiving_rank < nprocs; receiving_rank++) {
        MPI_Reduce(&my_counts[receiving_rank],
        &total_items_for_me,
        1,
        MPI_INT,
        MPI_SUM,
        receiving_rank,
        MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    std::cout << "Process " << rank << " needs to allocate space for "
        << total_items_for_me << " items"
        << std::endl;


    for (int prefix_count_index = 0; prefix_count_index < nprocs; prefix_count_index++) {
        MPI_Exscan(&my_counts[prefix_count_index],
        &my_prefix_counts[prefix_count_index],
        1,
        MPI_INT,
        MPI_SUM,
        MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
    }

    std::cout << "The prefix counts for process " << rank << " are "
        << my_prefix_counts[0] << ", "
        << my_prefix_counts[1] << ", "
        << my_prefix_counts[2]
        << std::endl;







}
