#include "mpi.h"
#include "mpi_prefix_sum.hpp"
#include <cmath>
#include <bits/stdc++.h>

// sum_val for process with rank 0 will be the sum of all my_val’s
void my_prefix_sum(int local_n, int *sum_matrix)
{
    // update sum_vals to be the prefix sum
    int myrank, nprocs; // add any other variables you need

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    int p = (int)round(std::sqrt(nprocs));

    int dimsize[2] = {p, p};
    int periodic[2] = {0, 0};
    MPI_Comm comm_2d;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dimsize, periodic, 0, &comm_2d);

    int my_proc_coords[2];
    MPI_Cart_coords(comm_2d, myrank, 2, my_proc_coords);
    int my_proc_row = my_proc_coords[0];
    int my_proc_col = my_proc_coords[1];

    // update sum_matrix to hold prefix sum for each row of each rank
    for (int local_row = 0; local_row < local_n; local_row++) {
        for (int local_col = 0; local_col < local_n; local_col++) {
            if (local_col != 0) {
                sum_matrix[local_row * local_n + local_col] +=
                sum_matrix[local_row * local_n + local_col - 1];
            }
        }
    }

    // complete prefix sum for each
    for (int local_col= 0; local_col < local_n; local_col++) {
        for (int local_row = 0; local_row < local_n; local_row++) {
            if (local_row != 0) {
                sum_matrix[local_row * local_n + local_col] +=
                sum_matrix[local_row * local_n + local_col - local_n];
            }
        }
    }

    MPI_Comm comm_row, comm_col;
    MPI_Comm_split(comm_2d, my_proc_row, my_proc_col, &comm_row);
    MPI_Comm_split(comm_2d, my_proc_col, my_proc_row, &comm_col);

    MPI_Barrier(MPI_COMM_WORLD);

    int row_comm_buff[local_n];
    int row_comm_storage[local_n];

    // should be able to replace this with fill
    for (int storage_row = 0; storage_row < local_n; storage_row++) {
        row_comm_storage[storage_row] = 0;
    }

    for (int sending_proc_col = 0; sending_proc_col < p - 1; sending_proc_col++) {
        if (sending_proc_col == my_proc_col) {
            for (int local_row = 0; local_row < local_n; local_row++) {
                row_comm_buff[local_row] = sum_matrix[local_row * local_n + local_n - 1];
            }
        }

        MPI_Bcast(row_comm_buff, local_n, MPI_INT, sending_proc_col, comm_row);
        MPI_Barrier(MPI_COMM_WORLD);

        if (my_proc_col > sending_proc_col) {
            for (int local_row = 0; local_row < local_n; local_row++) {
                row_comm_storage[local_row] += row_comm_buff[local_row];
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    for (int local_col = 0; local_col < local_n; local_col++) {
        for (int local_row = 0; local_row < local_n; local_row++) {
            sum_matrix[local_row * local_n + local_col] += row_comm_storage[local_row];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    int col_comm_buff[local_n];
    int col_comm_storage[local_n];

    // should be able to replace this with fill
    for (int storage_col = 0; storage_col < local_n; storage_col++) {
        col_comm_storage[storage_col] = 0;
    }

    for (int sending_proc_row = 0; sending_proc_row < p - 1; sending_proc_row++) {
        if (sending_proc_row == my_proc_row) {
            for (int local_col = 0; local_col < local_n; local_col++) {
                col_comm_buff[local_col] = sum_matrix[(local_n - 1) * local_n + local_col];
            }
        }

        MPI_Bcast(col_comm_buff, local_n, MPI_INT, sending_proc_row, comm_col);
        MPI_Barrier(MPI_COMM_WORLD);

        if (my_proc_row > sending_proc_row) {
            for (int local_col = 0; local_col < local_n; local_col++) {
                col_comm_storage[local_col] += col_comm_buff[local_col];
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    for (int local_row = 0; local_row < local_n; local_row++) {
            for (int local_col = 0; local_col < local_n; local_col++) {
                sum_matrix[local_row * local_n + local_col] += col_comm_storage[local_col];
            }
        }
}
