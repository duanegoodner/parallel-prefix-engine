#include "mpi_prefix_sum/mpi_cartesian_grid.hpp"

#include <cmath>

MpiCartesianGrid::MpiCartesianGrid(int rank, int size) {
  p_ = static_cast<int>(std::round(std::sqrt(size)));

  int dims[2] = {p_, p_};
  int periodic[2] = {0, 0};
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periodic, 0, &comm_2d_);

  int coords[2];
  MPI_Cart_coords(comm_2d_, rank, 2, coords);
  proc_row_ = coords[0];
  proc_col_ = coords[1];

  MPI_Comm_split(comm_2d_, proc_row_, proc_col_, &comm_row_);
  MPI_Comm_split(comm_2d_, proc_col_, proc_row_, &comm_col_);
}

MpiCartesianGrid::~MpiCartesianGrid() {
  MPI_Comm_free(&comm_row_);
  MPI_Comm_free(&comm_col_);
  MPI_Comm_free(&comm_2d_);
}
