// ----------------------------------------------------------------------------
// matrix_io.hpp
//
// Matrix io definitions.
// This header is part of the prefix sum project.
// ----------------------------------------------------------------------------

#pragma once

#include <mpi.h>

#include <vector>

#include "common/matrix_output.hpp"

template <typename T>
MPI_Datatype GetMpiDatatype();

template <>
inline MPI_Datatype GetMpiDatatype<int>() {
  return MPI_INT;
}

template <>
inline MPI_Datatype GetMpiDatatype<float>() {
  return MPI_FLOAT;
}

template <>
inline MPI_Datatype GetMpiDatatype<double>() {
  return MPI_DOUBLE;
}

template <typename T>
void PrintDistributedMatrix(
    int rank,
    int size,
    int num_rows,
    int num_cols,
    const std::vector<T> &local_mat,
    const std::string &header = ""
) {
  if (rank == 0) {
    if (!header.empty()) {
      PrintMatrix(header); // Print the header line before matrix output
    }
    PrintMatrix(FormatMatrix(rank, num_rows, num_cols, local_mat));

    std::vector<T> recv_buf(num_rows * num_cols);
    MPI_Status status;
    for (int i = 1; i < size; ++i) {
      MPI_Recv(
          recv_buf.data(),
          num_rows * num_cols,
          GetMpiDatatype<T>(),
          i,
          0,
          MPI_COMM_WORLD,
          &status
      );
      PrintMatrix(FormatMatrix(i, num_rows, num_cols, recv_buf));
    }
  } else {
    MPI_Send(
        local_mat.data(),
        num_rows * num_cols,
        GetMpiDatatype<T>(),
        0,
        0,
        MPI_COMM_WORLD
    );
  }
}