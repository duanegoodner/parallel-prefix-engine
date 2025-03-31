#pragma once

#include <mpi.h>

class MpiCartesianGrid {
public:
  MpiCartesianGrid(int rank, int size);
  ~MpiCartesianGrid();

  int proc_row() const { return proc_row_; }
  int proc_col() const { return proc_col_; }

  MPI_Comm cart_comm() const { return comm_2d_; }
  MPI_Comm row_comm() const { return comm_row_; }
  MPI_Comm col_comm() const { return comm_col_; }

  int grid_dim() const { return p_; }

private:
  int p_;
  int proc_row_;
  int proc_col_;

  MPI_Comm comm_2d_;
  MPI_Comm comm_row_;
  MPI_Comm comm_col_;
};
