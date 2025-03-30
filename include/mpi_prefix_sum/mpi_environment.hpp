#pragma once

#include <mpi.h>

class MpiEnvironment {
 public:
  MpiEnvironment(int& argc, char**& argv);
  ~MpiEnvironment();

  int rank() const { return rank_; }
  int size() const { return size_; }

 private:
  int rank_ = -1;
  int size_ = -1;
};
