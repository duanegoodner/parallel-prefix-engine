#pragma once

#include "common/prefix_sum_solver.hpp"
#include "mpi_prefix_sum/mpi_environment.hpp"
#include "mpi_prefix_sum/mpi_utils.hpp"

class MpiPrefixSumSolver : public PrefixSumSolver {
 public:
  MpiPrefixSumSolver(const MpiEnvironment& mpi, const ProgramArgs& args);
  void Compute(std::vector<int>& local_matrix) override;

 private:
  const MpiEnvironment& mpi_;
  const ProgramArgs& args_;
};
