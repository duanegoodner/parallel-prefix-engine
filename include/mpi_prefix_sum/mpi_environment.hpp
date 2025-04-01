// ----------------------------------------------------------------------------
// mpi_environment.hpp
//
// Mpi environment definitions.
// This header is part of the prefix sum project.
// ----------------------------------------------------------------------------

#pragma once

#include <mpi.h>

#include "common/program_args.hpp"

// Class MpiEnvironment: Encapsulates MPI initialization and finalization using
// RAII. Manages rank and size.
class MpiEnvironment {
public:
  // MpiEnvironment(int &argc, char **&argv);
  MpiEnvironment(const ProgramArgs &program_args);
  ~MpiEnvironment();

  int rank() const { return rank_; }
  int size() const { return size_; }

private:
  int rank_ = -1;
  int size_ = -1;
};