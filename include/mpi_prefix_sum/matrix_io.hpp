#ifndef MPI_PREFIX_SUM_MATRIX_IO_H_
#define MPI_PREFIX_SUM_MATRIX_IO_H_

#include <vector>
#include <mpi.h>
#include "common/matrix_output.hpp"

template <typename T>
MPI_Datatype GetMpiDatatype();

template <>
inline MPI_Datatype GetMpiDatatype<int>() { return MPI_INT; }

template <>
inline MPI_Datatype GetMpiDatatype<float>() { return MPI_FLOAT; }

template <>
inline MPI_Datatype GetMpiDatatype<double>() { return MPI_DOUBLE; }

template <typename T>
void PrintDistributedMatrix(int rank, int size, int local_n, const std::vector<T>& local_mat) {
  if (rank == 0) {
    PrintMatrix(FormatMatrix(rank, local_n, local_mat));
    std::vector<T> recv_buf(local_n * local_n);
    MPI_Status status;
    for (int i = 1; i < size; ++i) {
      MPI_Recv(recv_buf.data(), local_n * local_n, GetMpiDatatype<T>(), i, 0, MPI_COMM_WORLD, &status);
      PrintMatrix(FormatMatrix(i, local_n, recv_buf));
    }
  } else {
    MPI_Send(local_mat.data(), local_n * local_n, GetMpiDatatype<T>(), 0, 0, MPI_COMM_WORLD);
  }
}

#endif  // MPI_PREFIX_SUM_MATRIX_IO_H_