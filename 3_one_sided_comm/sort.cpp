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

    // std::cout << "Process " << rank << "'s key_value counts are "
    //     << my_counts[0] << ", "
    //     << my_counts[1] << ", "
    //     << my_counts[2]
    //     << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

    item** outgoing_buffer[nprocs];

    for (int dest_proc = 0; dest_proc < nprocs; dest_proc++) {
        int outgoing_counts = 0;
        outgoing_buffer[dest_proc] = (item**) malloc(my_counts[dest_proc] * sizeof(item*));
        for (int my_item_index = 0; my_item_index < N; my_item_index++) {
            if (myItems[my_item_index].key == dest_proc) {
                outgoing_buffer[dest_proc][outgoing_counts] = &myItems[my_item_index];
                outgoing_counts++;
            }
        }
    }

    int my_prefix_counts[nprocs];
    for (int prefix_count_index = 0; prefix_count_index < nprocs; prefix_count_index++) {
        my_prefix_counts[prefix_count_index] = 0;
    }

    for (int receiving_rank = 0; receiving_rank < nprocs; receiving_rank++) {
        MPI_Reduce(&my_counts[receiving_rank],
        nOut,
        1,
        MPI_INT,
        MPI_SUM,
        receiving_rank,
        MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (int prefix_count_index = 0; prefix_count_index < nprocs; prefix_count_index++) {
        MPI_Exscan(&my_counts[prefix_count_index],
        &my_prefix_counts[prefix_count_index],
        1,
        MPI_INT,
        MPI_SUM,
        MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
    }

    item * result = (item *) malloc(*nOut * sizeof(item));

    // directly copy items from self (no MPI_Put needed)
    int self_copy_dest_idx = my_prefix_counts[rank];
    for (int source_buf_idx = 0; source_buf_idx < my_counts[rank]; source_buf_idx++) {
        result[self_copy_dest_idx + source_buf_idx] = *outgoing_buffer[rank][source_buf_idx];
    }

    MPI_Datatype item_type;
    int lengths[2] = {1, 1};
    MPI_Aint displacements[2];
    struct item dummy_item;
    MPI_Aint base_address;
    MPI_Get_address(&dummy_item, &base_address);
    MPI_Get_address(&dummy_item.key, &displacements[0]);
    MPI_Get_address(&dummy_item.value, &displacements[1]);
    displacements[0] = MPI_Aint_diff(displacements[0], base_address);
    displacements[1] = MPI_Aint_diff(displacements[1], base_address);

    MPI_Datatype types[2] = {MPI_INT, MPI_INT};
    MPI_Type_create_struct(2, lengths, displacements, types, &item_type);
    MPI_Type_commit(&item_type);

    MPI_Win win;

    MPI_Win_create(
        result,
        *nOut * (int) sizeof(item),
        (int) sizeof(item),
        MPI_INFO_NULL,
        MPI_COMM_WORLD,
        &win);

    MPI_Win_fence(0, win);

    for (int dest_proc = 0; dest_proc < nprocs; dest_proc++) {
        if ((dest_proc != rank)) {
            int dest_index = my_prefix_counts[dest_proc];
            for (int outgoing_index = 0; outgoing_index < my_counts[dest_proc]; outgoing_index++) {
                MPI_Put(
                    outgoing_buffer[dest_proc][outgoing_index],
                    1,
                    item_type,
                    dest_proc,
                    my_prefix_counts[dest_proc] + outgoing_index,
                    1,
                    item_type,
                    win
                );
            }

        MPI_Barrier(MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Win_fence(0, win);

    for (int result_idx = 0; result_idx < *nOut; result_idx++) {
        std::cout << "Proces " << rank << " has item with index = "
            << result[result_idx].key << " as its element #" << result_idx
            << std::endl;
    }

    *myResult = result;
}
