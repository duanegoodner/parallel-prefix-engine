// ----------------------------------------------------------------------------
// mpi_cartesian_grid.hpp
//
// Mpi cartesian grid definitions.
// This header is part of the prefix sum project.
// ----------------------------------------------------------------------------

#pragma once

#include <mpi.h>

// Class MpiCartesianGrid:
class MpiCartesianGrid {
public:
  MpiCartesianGrid(int rank, int num_rows, int num_cols);
  ~MpiCartesianGrid();

  int proc_row() const { return proc_row_; }
  int proc_col() const { return proc_col_; }

  MPI_Comm cart_comm() const { return comm_2d_; }
  MPI_Comm row_comm() const { return comm_row_; }
  MPI_Comm col_comm() const { return comm_col_; }

  // int grid_dim() const { return p_; }
  int rank() const { return rank_; }
  int num_rows() const { return num_rows_; }
  int num_cols() const { return num_cols_; }
  int size() const { return num_rows_ * num_cols_; }

private:
  // int p_;
  int rank_;
  int num_rows_;
  int num_cols_;
  int proc_row_;
  int proc_col_;

  MPI_Comm comm_2d_;
  MPI_Comm comm_row_;
  MPI_Comm comm_col_;
};