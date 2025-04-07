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
