#ifndef HELPER_
#define HELPER_
#include <string>
#include "mpi.h"

typedef int val;

struct item {
    int key;
    val value;
};

std::string to_string(item v) {
    // std::string str = std::to_string(v.key) + ":" + std::to_string(v.value);
    std::string str = std::to_string(v.key);
    return str;
}

bool item_cmp(item const &lhs, item const &rhs) {
    return lhs.key < rhs.key;
    return ((lhs.key == rhs.key) && lhs.value < rhs.value) || lhs.key < rhs.key;
}

void print_local_vals(int rank, int local_n, item *values)
{
    std::string output = "rank " + std::to_string(rank) + ": \n";
    for (int i = 0; i < local_n; i++)
    {
        output += "\t" + to_string(values[i]);
    }
    fprintf(stdout, "%s\n", output.c_str());
    fflush(stdout);
}

void print_global_vals(int rank, int procs, int local_n, item *values)
{
    // NOTE: sending everything to rank 0 to print is not efficient in terms of scalability but ensures everything will be printed and flushed in order. This is a debugging tool and not a performance level tool.
    if (rank == 0)
    {
        print_local_vals(rank, local_n, values);
        int *counts = (int *)malloc(sizeof(int) * procs);
        counts[0] = 0;
        int max_count = 0;
        for (int i = 1; i < procs; i++)
        {
            counts[i] = 0;
            MPI_Recv(&counts[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if(counts[i] > max_count) max_count = counts[i];
        }
        item *recv_val = (item *)malloc(sizeof(item) * max_count);
        for (int i = 1; i < procs; i++)
        {
            MPI_Recv(recv_val, counts[i] * sizeof(item), MPI_BYTE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            print_local_vals(i, counts[i], recv_val);
        }
        free(recv_val);
        free(counts);
    }
    else
    {
        MPI_Send(&local_n, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&values[0], local_n * sizeof(item), MPI_BYTE, 0, 1, MPI_COMM_WORLD);
    }
}

#endif
