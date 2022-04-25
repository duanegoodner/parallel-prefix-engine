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

    // get quantity of each key type
    int my_counts[nprocs];
    std::fill(my_counts, my_counts + nprocs, 0);
    for (auto my_item_index = 0; my_item_index < N; my_item_index++) {
        my_counts[myItems[my_item_index].key] += 1;
    }

    // copy each item to appropriate outgoing buffer
    item** outgoing_buffers[nprocs];
    for (int dest_proc = 0; dest_proc < nprocs; dest_proc++) {
        int outgoing_counts = 0;
        outgoing_buffers[dest_proc] = (item**) malloc(my_counts[dest_proc] * sizeof(item*));
        for (int my_item_index = 0; my_item_index < N; my_item_index++) {
            if (myItems[my_item_index].key == dest_proc) {
                outgoing_buffers[dest_proc][outgoing_counts] = &myItems[my_item_index];
                outgoing_counts++;
            }
        }
    }

    // each process gets total number of items with key == its rank value
    for (int receiving_rank = 0; receiving_rank < nprocs; receiving_rank++) {
        MPI_Reduce(&my_counts[receiving_rank],
        nOut,
        1,
        MPI_INT,
        MPI_SUM,
        receiving_rank,
        MPI_COMM_WORLD);
    }

    /*
    Each process learns # of items of each key val in lower rank prcesses. Will
    used this info to know where to place items in results buffers.
     */
    int my_prefix_counts[nprocs];
    std::fill(my_prefix_counts, my_prefix_counts + nprocs, 0);
    for (int prefix_count_index = 0; prefix_count_index < nprocs; prefix_count_index++) {
        MPI_Exscan(&my_counts[prefix_count_index],
        &my_prefix_counts[prefix_count_index],
        1,
        MPI_INT,
        MPI_SUM,
        MPI_COMM_WORLD);
    }

    // allocate memory for storing all items with key == rank
    item * result = (item *) malloc(*nOut * sizeof(item));

    // items already on processor get directly copied to result memory
    int self_copy_dest_idx = my_prefix_counts[rank];
    for (int source_buf_idx = 0; source_buf_idx < my_counts[rank]; source_buf_idx++) {
        result[self_copy_dest_idx + source_buf_idx] = *outgoing_buffers[rank][source_buf_idx];
    }

    // custom MPI Datatype to be used during MPI_Put
    MPI_Datatype item_type;
    MPI_Type_contiguous(2, MPI_INT, &item_type);
    MPI_Type_commit(&item_type);

    // expose result memory for one-sided communication
    MPI_Win win;
    MPI_Win_create(
        result,
        *nOut * (int) sizeof(item),
        (int) sizeof(item),
        MPI_INFO_NULL,
        MPI_COMM_WORLD,
        &win);

    MPI_Win_fence(0, win);

    // Couldn't get multi-item Put to work, so use loop and Put one at a time
    for (int dest_proc = 0; dest_proc < nprocs; dest_proc++) {
        if ((dest_proc != rank)) {
            int dest_index = my_prefix_counts[dest_proc];
            for (int outgoing_index = 0; outgoing_index < my_counts[dest_proc]; outgoing_index++) {
                MPI_Put(
                    outgoing_buffers[dest_proc][outgoing_index],
                    1,
                    item_type,
                    dest_proc,
                    my_prefix_counts[dest_proc] + outgoing_index,
                    1,
                    item_type,
                    win
                );
            }
        }
    }

    MPI_Win_fence(0, win);

    *myResult = result;
}
