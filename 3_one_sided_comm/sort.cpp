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

                // std::cout << "Process " << rank << " just add item with key value "
                //     << outgoing_buffer[dest_proc][outgoing_counts]->key << " to its buffer for "
                //     << dest_proc << std::endl;

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

    std::cout << "Process " << rank << " needs to allocate space for "
        << *nOut << " items"
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

    // std::cout << "The prefix counts for process " << rank << " are "
    //     << my_prefix_counts[0] << ", "
    //     << my_prefix_counts[1] << ", "
    //     << my_prefix_counts[2]
    //     << std::endl;


    item * result = (item *) malloc(*nOut * sizeof(item));
    //   std::cout << "Process " << rank << " allocated space for " << *nOut << " items in result"
    //     << std::endl;

    // directly copy items from self (no MPI_Put needed)
    int self_copy_dest_idx = my_prefix_counts[rank];
    for (int source_buf_idx = 0; source_buf_idx < my_counts[rank]; source_buf_idx++) {
        result[self_copy_dest_idx + source_buf_idx] = *outgoing_buffer[rank][source_buf_idx];
    }

    // check what's in results now
    // for (int result_idx = 0; result_idx < *nOut; result_idx++) {
    //     std::cout << "Process " << rank << " has item with key = "
    //         << result[result_idx].key << " at index " << result_idx
    //         << std::endl;
    // }

    // myResult = (item **) malloc(*nOut * sizeof(item));
    // std::cout << "Process " << rank << " allocated space for " << *nOut << " items in myResult"
    //     << std::endl;


    // debug print OUTGOING BUFFER KEYS
    // for (int dest_proc = 0; dest_proc < nprocs; dest_proc++) {
    //     for (int outgoing_index = 0; outgoing_index < my_counts[dest_proc]; outgoing_index++) {
    //         std::cout << "Proces  " << rank << " has item with key "
    //             << outgoing_buffer[dest_proc][outgoing_index]->key
    //             << " for process " << dest_proc << std::endl;
    //     }
    // }


    // SELF COPY WITH DOUBLE POINTER RESULT
    // int first_dest_index = my_prefix_counts[rank];
    // for (int self_copy_buf_idx = 0; self_copy_buf_idx < my_counts[rank]; self_copy_buf_idx++) {
    //     myResult[first_dest_index + self_copy_buf_idx] = outgoing_buffer[rank][self_copy_buf_idx];
    //     std::cout << "Process " << rank << " just placed item with key = "
    //         << myResult[first_dest_index + self_copy_buf_idx]->key << " at its Results index "
    //         << first_dest_index + self_copy_buf_idx << std::endl;

    // }

    MPI_Datatype item_type;
    MPI_Type_contiguous(2, MPI_INT, &item_type);
    MPI_Type_commit(&item_type);

    MPI_Win win;

    MPI_Win_create(
        result,
        *nOut * (int) sizeof(item),
        (int) sizeof(item),
        MPI_INFO_NULL,
        MPI_COMM_WORLD,
        &win);

    // for (int result_index = 0; result_index < *nOut; result_index++) {
    //     std::cout <<  result[result_index].key << std::endl;
    // }

    MPI_Win_fence(0, win);


    for (int dest_proc = 0; dest_proc < nprocs; dest_proc++) {
        if ((dest_proc != rank) & (my_counts[dest_proc] != 0)) {
            std::cout << " Process " << rank << " has data to put in process "
                << dest_proc << std::endl;

            MPI_Put(
                *outgoing_buffer[dest_proc],
                my_counts[dest_proc],
                item_type,
                dest_proc,
                my_prefix_counts[dest_proc],
                my_counts[dest_proc],
                item_type,
                win);
        }
    }

    MPI_Win_fence(0, win);

    for (int result_idx = 0; result_idx < *nOut; result_idx++) {
        std::cout << "Proces " << rank << " has item with index = "
            << result[result_idx].key << " as its element #" << result_idx
            << std::endl;
    }
}

    // MPI_Win_fence(0, win);
    // MPI_Barrier(MPI_COMM_WORLD);




    // if (rank == 0) {
    //     std::cout << "Process " << rank << " has key value of " <<
    //         myResult[0]->key << " at index 0" << std::endl;
    // }

    //     if (rank == 0) {
    //     std::cout << "Process " << rank << " has key value of " <<
    //         myResult[1]->key << " at index 1" << std::endl;
    // }

    //     if (rank == 0) {
    //     std::cout << "Process " << rank << " has key value of " <<
    //         myResult[2]->key << " at index 2" << std::endl;
    // }




// }
