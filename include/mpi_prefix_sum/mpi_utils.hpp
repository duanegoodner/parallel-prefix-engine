#pragma once


void print_local_mat(int rank, int local_n, int *local_mat);
void print_global_mat(int rank, int procs, int local_n, int *local_mat);
